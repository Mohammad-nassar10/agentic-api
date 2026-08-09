//! Per-session prompt prefix stored in the database.

use super::super::pool::{DbPool, DbResult};

/// A folded prompt prefix for one session.
///
/// Maps to the `session_prefix` table. `replacement` is sent in place of the
/// first `replaced_count` messages of an incoming request.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SessionPrefix {
    /// Session identifier, taken from the request's session header.
    pub session_id: String,

    /// How many leading client messages `replacement` replaces.
    pub replaced_count: i64,

    /// Messages sent in their place, as a JSON array string.
    pub replacement: String,
}

/// Look up the stored prefix for a session.
///
/// # Errors
/// Returns `DbResult::Err` if the query fails.
pub async fn get(pool: &DbPool, session_id: &str) -> DbResult<Option<SessionPrefix>> {
    sqlx::query_as::<_, SessionPrefix>("SELECT * FROM session_prefix WHERE session_id = $1")
        .bind(session_id)
        .fetch_optional(pool)
        .await
}

/// Insert or replace the stored prefix for a session.
///
/// # Errors
/// Returns `DbResult::Err` if the write fails.
pub async fn upsert(pool: &DbPool, session_id: &str, replaced_count: i64, replacement: &str) -> DbResult<()> {
    sqlx::query(
        "INSERT INTO session_prefix (session_id, replaced_count, replacement) \
         VALUES ($1, $2, $3) \
         ON CONFLICT (session_id) DO UPDATE SET \
         replaced_count = EXCLUDED.replaced_count, \
         replacement = EXCLUDED.replacement",
    )
    .bind(session_id)
    .bind(replaced_count)
    .bind(replacement)
    .execute(pool)
    .await?;
    Ok(())
}
