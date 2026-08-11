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
