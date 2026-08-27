//! Backend load reported by llm-d on a response.
//!
//! llm-d's `metrics-to-headers` endpoint-picker plugin stamps the serving
//! endpoint's latest metrics onto the response it returns. Reading them here
//! gives the gateway an in-band view of backend pressure without scraping
//! Prometheus or knowing anything about the pool.
//!
//! Currently observed only. The point of collecting it is to eventually fold a
//! session's history on backend pressure rather than on every turn.

use std::time::Duration;

use agentic_core::error::Error;
use http::HeaderMap;

const KV_CACHE_HEADER: &str = "x-llm-d-kv-cache-utilization";
const WAITING_QUEUE_HEADER: &str = "x-llm-d-waiting-queue";
const RUNNING_REQUESTS_HEADER: &str = "x-llm-d-running-requests";
const METRICS_AGE_HEADER: &str = "x-llm-d-metrics-age-ms";

#[derive(Debug, Default, Clone, PartialEq)]
pub struct PoolSignals {
    /// Fraction in `[0,1]`, despite the upstream metric's "percent" name.
    pub kv_cache_utilization: Option<f64>,
    pub waiting_queue: Option<u64>,
    pub running_requests: Option<u64>,
    /// Age of the metrics snapshot. Absent when the endpoint was never scraped.
    pub age: Option<Duration>,
}

impl PoolSignals {
    /// Read the signals from a response.
    ///
    /// `None` means the response carried none of these headers — an endpoint
    /// picker without the plugin, or an upstream that is not llm-d. That is
    /// deliberately distinct from a reading of zero, which means an idle pool.
    #[must_use]
    pub fn from_headers(headers: &HeaderMap) -> Option<Self> {
        let signals = Self {
            kv_cache_utilization: parse_header(headers, KV_CACHE_HEADER),
            waiting_queue: parse_header(headers, WAITING_QUEUE_HEADER),
            running_requests: parse_header(headers, RUNNING_REQUESTS_HEADER),
            age: parse_header::<u64>(headers, METRICS_AGE_HEADER).map(Duration::from_millis),
        };
        (signals != Self::default()).then_some(signals)
    }
}

/// A malformed value is dropped rather than failing the read: these are
/// advisory signals, and one bad header should not discard the others.
fn parse_header<T: std::str::FromStr>(headers: &HeaderMap, name: &str) -> Option<T> {
    headers.get(name)?.to_str().ok()?.trim().parse().ok()
}

pub const KV_CACHE_ENV: &str = "COMPACTION_KV_CACHE_THRESHOLD";
pub const WAITING_QUEUE_ENV: &str = "COMPACTION_WAITING_QUEUE_THRESHOLD";
pub const RUNNING_REQUESTS_ENV: &str = "COMPACTION_RUNNING_REQUESTS_THRESHOLD";
pub const MAX_AGE_ENV: &str = "COMPACTION_MAX_METRICS_AGE_MS";

/// Levels of backend pressure at or above which a session is worth compacting.
///
/// Each is independent and optional; an unset one is not consulted. With none
/// set the gateway compacts on every turn, which is the behaviour when the
/// serving endpoint reports nothing at all.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Thresholds {
    pub kv_cache_utilization: Option<f64>,
    pub waiting_queue: Option<u64>,
    pub running_requests: Option<u64>,
    /// Snapshots older than this are ignored entirely. Not a trigger — a guard
    /// against deciding on metrics that no longer describe the endpoint.
    pub max_age: Option<Duration>,
}

impl Thresholds {
    /// Read the thresholds from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if a variable is set but unparseable, or if the
    /// KV cache threshold falls outside `[0,1]`. A silently ignored threshold
    /// would read as "compaction never triggers" with nothing to explain it.
    pub fn from_env() -> Result<Self, Error> {
        let kv_cache_utilization = parse_env::<f64>(KV_CACHE_ENV)?;
        if let Some(threshold) = kv_cache_utilization
            && !(0.0..=1.0).contains(&threshold)
        {
            return Err(Error::Config(format!(
                "{KV_CACHE_ENV} must be a fraction between 0 and 1, got {threshold}"
            )));
        }

        Ok(Self {
            kv_cache_utilization,
            waiting_queue: parse_env(WAITING_QUEUE_ENV)?,
            running_requests: parse_env(RUNNING_REQUESTS_ENV)?,
            max_age: parse_env::<u64>(MAX_AGE_ENV)?.map(Duration::from_millis),
        })
    }

    fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Whether the reported load meets any configured threshold.
    ///
    /// Disjunctive by design: a saturated cache and a growing queue each justify
    /// compaction on their own, so requiring both would almost never fire.
    fn met_by(&self, signals: &PoolSignals) -> bool {
        let over = |threshold: Option<f64>, reading: Option<f64>| match (threshold, reading) {
            (Some(threshold), Some(reading)) => reading >= threshold,
            // A metric the endpoint did not report cannot trigger its threshold,
            // but the other thresholds are still worth checking.
            _ => false,
        };

        over(self.kv_cache_utilization, signals.kv_cache_utilization)
            || over(cast(self.waiting_queue), cast(signals.waiting_queue))
            || over(cast(self.running_requests), cast(signals.running_requests))
    }
}

/// Whether this turn should be folded into the session prefix.
///
/// Fails open — no thresholds configured, no signals, or a stale snapshot all
/// compact. Compacting needlessly costs one background call to the compaction
/// service; skipping it costs prompt tokens on every subsequent turn.
#[must_use]
pub fn should_compact(thresholds: &Thresholds, signals: Option<&PoolSignals>) -> bool {
    if thresholds.is_empty() {
        return true;
    }
    let Some(signals) = signals else {
        return true;
    };
    match (thresholds.max_age, signals.age) {
        // Too old to describe the endpoint we are deciding about.
        (Some(max_age), Some(age)) if age > max_age => return true,
        _ => {}
    }
    thresholds.met_by(signals)
}

/// Widening only; every value here originates as a header count or a small
/// configured limit, far below f64's exact-integer range.
#[expect(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
fn cast(value: Option<u64>) -> Option<f64> {
    value.map(|value| value as f64)
}

/// Unset means "not configured"; set-but-invalid is a configuration error.
fn parse_env<T: std::str::FromStr>(name: &str) -> Result<Option<T>, Error>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => value
            .trim()
            .parse()
            .map(Some)
            .map_err(|error| Error::Config(format!("{name} is not a valid number: {error}"))),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(Error::Config(format!("failed to read {name}: {error}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        headers
    }

    #[test]
    fn reads_the_signals_llm_d_sends() {
        let signals = PoolSignals::from_headers(&headers(&[
            (KV_CACHE_HEADER, "0.8125"),
            (WAITING_QUEUE_HEADER, "7"),
            (RUNNING_REQUESTS_HEADER, "3"),
            (METRICS_AGE_HEADER, "460"),
        ]))
        .expect("headers were present");

        assert_eq!(signals.kv_cache_utilization, Some(0.8125));
        assert_eq!(signals.waiting_queue, Some(7));
        assert_eq!(signals.running_requests, Some(3));
        assert_eq!(signals.age, Some(Duration::from_millis(460)));
    }

    /// An upstream without the plugin must be distinguishable from an idle pool.
    #[test]
    fn absent_headers_are_not_a_reading_of_zero() {
        assert_eq!(
            PoolSignals::from_headers(&headers(&[("content-type", "application/json")])),
            None
        );

        let idle = PoolSignals::from_headers(&headers(&[(KV_CACHE_HEADER, "0.0000")])).expect("header was present");
        assert_eq!(idle.kv_cache_utilization, Some(0.0));
    }

    /// The age header is omitted for an endpoint that was never scraped, and the
    /// remaining values are still worth reporting.
    fn signals(kv_cache: f64, queue: u64) -> PoolSignals {
        PoolSignals {
            kv_cache_utilization: Some(kv_cache),
            waiting_queue: Some(queue),
            running_requests: Some(0),
            age: Some(Duration::from_millis(5)),
        }
    }

    /// Without thresholds the gateway folds every turn, as it did before any of
    /// this existed.
    #[test]
    fn no_thresholds_always_compacts() {
        let none = Thresholds::default();
        assert!(should_compact(&none, Some(&signals(0.0, 0))));
        assert!(should_compact(&none, None));
    }

    #[test]
    fn a_threshold_gates_on_the_reading() {
        let thresholds = Thresholds {
            kv_cache_utilization: Some(0.8),
            ..Thresholds::default()
        };
        assert!(!should_compact(&thresholds, Some(&signals(0.79, 0))));
        assert!(should_compact(&thresholds, Some(&signals(0.80, 0))), "at the threshold");
        assert!(should_compact(&thresholds, Some(&signals(0.81, 0))));
    }

    /// Disjunctive: a calm cache must not mask a growing queue.
    #[test]
    fn any_single_threshold_is_enough() {
        let thresholds = Thresholds {
            kv_cache_utilization: Some(0.8),
            waiting_queue: Some(10),
            ..Thresholds::default()
        };
        assert!(should_compact(&thresholds, Some(&signals(0.1, 10))), "queue alone");
        assert!(should_compact(&thresholds, Some(&signals(0.9, 0))), "cache alone");
        assert!(!should_compact(&thresholds, Some(&signals(0.1, 0))), "neither");
    }

    /// An endpoint picker without the plugin reports nothing. Reading that as an
    /// idle pool would silently disable compaction.
    #[test]
    fn absent_signals_compact_rather_than_read_as_idle() {
        let thresholds = kv(0.8);
        assert!(should_compact(&thresholds, None));

        let unreported = PoolSignals {
            waiting_queue: Some(0),
            ..PoolSignals::default()
        };
        assert!(
            !should_compact(&thresholds, Some(&unreported)),
            "a reported metric below its threshold still gates"
        );
    }

    /// `max_age` guards the decision rather than triggering it: a snapshot too
    /// old to describe the endpoint is no basis for skipping a fold.
    #[test]
    fn a_stale_snapshot_is_not_trusted() {
        let thresholds = Thresholds {
            kv_cache_utilization: Some(0.8),
            max_age: Some(Duration::from_secs(1)),
            ..Thresholds::default()
        };

        let fresh = PoolSignals {
            age: Some(Duration::from_millis(10)),
            ..signals(0.1, 0)
        };
        assert!(!should_compact(&thresholds, Some(&fresh)), "fresh and idle: skip");

        let stale = PoolSignals {
            age: Some(Duration::from_secs(30)),
            ..signals(0.1, 0)
        };
        assert!(should_compact(&thresholds, Some(&stale)), "too old to act on");
    }

    fn kv(threshold: f64) -> Thresholds {
        Thresholds {
            kv_cache_utilization: Some(threshold),
            ..Thresholds::default()
        }
    }

    #[test]
    fn a_bad_or_missing_value_does_not_discard_the_others() {
        let signals = PoolSignals::from_headers(&headers(&[
            (KV_CACHE_HEADER, "not-a-number"),
            (WAITING_QUEUE_HEADER, "7"),
        ]))
        .expect("headers were present");

        assert_eq!(signals.kv_cache_utilization, None);
        assert_eq!(signals.waiting_queue, Some(7));
        assert_eq!(signals.age, None);
    }
}
