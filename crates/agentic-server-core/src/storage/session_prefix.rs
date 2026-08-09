//! Session prefix storage operations.

use std::sync::Arc;

use serde_json::Value;

use super::models::session_prefix;
use super::pool::DbPool;
use super::types::{StorageError, StoreResult};
use crate::utils::common::{deserialize_from_str, serialize_to_string};

/// A session's folded prefix, ready to substitute into an incoming request.
///
/// Substitution drops the first `replaced_count` messages of the request and
/// puts `replacement` in front of the rest.
///
/// This trusts the client to resend the same leading messages. Validating that
/// assumption — by storing a hash of the replaced messages and comparing on
/// arrival, so a mismatch forwards the request untouched — is left for later.
#[derive(Debug, Clone)]
pub struct SessionPrefixData {
    /// How many leading client messages `replacement` replaces.
    pub replaced_count: usize,

    /// Messages sent in place of those.
    pub replacement: Vec<Value>,
}

/// Session prefix storage operations.
#[derive(Clone, Debug)]
pub struct SessionPrefixStore {
    pool: Option<Arc<DbPool>>,
}

impl SessionPrefixStore {
    /// Creates a disabled store (no persistence).
    #[must_use]
    pub fn disabled() -> Self {
        Self { pool: None }
    }

    /// Creates a new store with a database pool.
    #[must_use]
    pub fn new(pool: Arc<DbPool>) -> Self {
        Self { pool: Some(pool) }
    }

    fn pool(&self) -> StoreResult<&DbPool> {
        self.pool.as_deref().ok_or(StorageError::NotConfigured)
    }

    /// Fetches the stored prefix for a session, or `None` when it has no entry.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the query fails, the stored replacement is
    /// not valid JSON, or the store is disabled.
    pub async fn get(&self, session_id: &str) -> StoreResult<Option<SessionPrefixData>> {
        let Some(row) = session_prefix::get(self.pool()?, session_id).await? else {
            return Ok(None);
        };
        Ok(Some(SessionPrefixData {
            // Counts are written from message-array lengths; a negative or
            // oversized value means a corrupted row, treated as replacing nothing.
            replaced_count: usize::try_from(row.replaced_count).unwrap_or_default(),
            replacement: deserialize_from_str(&row.replacement)?,
        }))
    }

    /// Inserts or replaces the stored prefix for a session.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the replacement cannot be serialized, the
    /// write fails, or the store is disabled.
    pub async fn upsert(&self, session_id: &str, replaced_count: usize, replacement: &[Value]) -> StoreResult<()> {
        let replacement = serialize_to_string(&replacement)?;
        // Counts come from message-array lengths, which cannot realistically
        // exceed i64; saturate rather than fail a background write.
        let replaced_count = i64::try_from(replaced_count).unwrap_or(i64::MAX);
        session_prefix::upsert(self.pool()?, session_id, replaced_count, &replacement).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disabled_store_reports_not_configured() {
        let store = SessionPrefixStore::disabled();

        assert!(store.get("s-1").await.is_err());
        assert!(store.upsert("s-1", 2, &[]).await.is_err());
    }
}
