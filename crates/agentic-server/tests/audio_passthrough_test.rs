//! `/v1/audio/transcriptions` and `/v1/audio/translations` stream the client's multipart upload
//! to the upstream as it arrives, within the request-size limit, and relay the answer unchanged.
#[allow(dead_code)]
mod common;

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Bytes;
use axum::extract::Request;
use axum::response::IntoResponse;
use axum::routing::post;
use common::{spawn_gateway, test_config, test_state, test_state_with_max_request_body_size};
use http::{HeaderMap, StatusCode};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

const UPLOAD_CONTENT_TYPE: &str = "multipart/form-data; boundary=agentic-audio-boundary";

/// A multipart upload with a binary audio part, so the forwarded body can be compared byte
/// for byte against what the upstream received.
const UPLOAD: &[u8] = b"--agentic-audio-boundary\r\n\
Content-Disposition: form-data; name=\"model\"\r\n\r\n\
whisper-1\r\n\
--agentic-audio-boundary\r\n\
Content-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n\
Content-Type: audio/wav\r\n\r\n\
RIFF\x00\x01\x02\xff\xfe audio\r\n\
--agentic-audio-boundary--\r\n";

/// Irregular whitespace and an unknown field, so the relayed body can be compared byte for byte.
const TRANSCRIPTION: &str = r#"{"text":"hello world",  "unknown_field":{"kept":true}}"#;

const TRANSCRIPTION_SSE: &str = concat!(
    "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello\"}\n\n",
    "data: [DONE]\n\n"
);

/// What the upstream received for the last request.
#[derive(Default)]
struct Received {
    path: String,
    headers: HeaderMap,
    body: Bytes,
}

/// Upstream audio endpoints that answer with a fixed response and record what they were sent.
#[derive(Clone)]
struct Upstream {
    status: StatusCode,
    content_type: &'static str,
    body: &'static str,
    received: Arc<Mutex<Received>>,
    calls: Arc<AtomicUsize>,
}

impl Upstream {
    fn new(status: StatusCode, content_type: &'static str, body: &'static str) -> Self {
        Self {
            status,
            content_type,
            body,
            received: Arc::default(),
            calls: Arc::default(),
        }
    }

    async fn spawn(&self) -> String {
        let upstream = self.clone();
        let handler = move |req: Request| {
            let upstream = upstream.clone();
            async move {
                let (parts, body) = req.into_parts();
                // A cut-off upload ends early; record whatever arrived.
                let body = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
                *upstream.received.lock().await = Received {
                    path: parts.uri.path().to_owned(),
                    headers: parts.headers,
                    body,
                };
                upstream.calls.fetch_add(1, Ordering::SeqCst);
                let content_type = [(http::header::CONTENT_TYPE, upstream.content_type)];
                (upstream.status, content_type, upstream.body).into_response()
            }
        };
        let app = Router::new()
            .route("/v1/audio/transcriptions", post(handler.clone()))
            .route("/v1/audio/translations", post(handler));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }
}

async fn gateway(upstream_url: &str) -> String {
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(upstream_url))).await;
    gateway_url
}

async fn gateway_with_limit(upstream_url: &str, limit: usize) -> String {
    let limit = NonZeroUsize::new(limit).unwrap();
    let (gateway_url, _gateway) =
        spawn_gateway(test_state_with_max_request_body_size(&test_config(upstream_url), limit)).await;
    gateway_url
}

/// The upload split into chunks and sent without a declared length.
fn chunked_upload() -> reqwest::Body {
    let chunks = UPLOAD
        .chunks(16)
        .map(|chunk| Ok::<_, std::io::Error>(Bytes::copy_from_slice(chunk)))
        .collect::<Vec<_>>();
    reqwest::Body::wrap_stream(futures::stream::iter(chunks))
}

async fn upload(url: String, body: impl Into<reqwest::Body>) -> reqwest::Response {
    reqwest::Client::new()
        .post(url)
        .header(http::header::CONTENT_TYPE, UPLOAD_CONTENT_TYPE)
        .body(body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn transcription_upload_is_forwarded_unchanged() {
    let upstream = Upstream::new(StatusCode::OK, "application/json", TRANSCRIPTION);
    let gateway_url = gateway(&upstream.spawn().await).await;

    let response = upload(format!("{gateway_url}/v1/audio/transcriptions"), UPLOAD).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), TRANSCRIPTION);
    let received = upstream.received.lock().await;
    assert_eq!(received.path, "/v1/audio/transcriptions");
    assert_eq!(&received.body[..], UPLOAD);
    assert_eq!(received.headers[http::header::CONTENT_TYPE], UPLOAD_CONTENT_TYPE);
    assert_eq!(received.headers[http::header::CONTENT_LENGTH], UPLOAD.len().to_string());
}

#[tokio::test]
async fn translation_is_forwarded_to_its_own_path() {
    let upstream = Upstream::new(StatusCode::OK, "application/json", TRANSCRIPTION);
    let gateway_url = gateway(&upstream.spawn().await).await;

    let response = upload(format!("{gateway_url}/v1/audio/translations"), UPLOAD).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), TRANSCRIPTION);
    let received = upstream.received.lock().await;
    assert_eq!(received.path, "/v1/audio/translations");
    assert_eq!(&received.body[..], UPLOAD);
}

#[tokio::test]
async fn streamed_transcription_is_relayed_as_sse() {
    let upstream = Upstream::new(StatusCode::OK, "text/event-stream", TRANSCRIPTION_SSE);
    let gateway_url = gateway(&upstream.spawn().await).await;

    let response = upload(format!("{gateway_url}/v1/audio/transcriptions"), UPLOAD).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[http::header::CONTENT_TYPE], "text/event-stream");
    assert_eq!(response.text().await.unwrap(), TRANSCRIPTION_SSE);
}

#[tokio::test]
async fn upstream_errors_are_forwarded_unchanged() {
    const ERROR: &str = r#"{"error":{"message":"unsupported audio format","type":"BadRequestError"}}"#;
    let upstream = Upstream::new(StatusCode::BAD_REQUEST, "application/json", ERROR);
    let gateway_url = gateway(&upstream.spawn().await).await;

    let response = upload(format!("{gateway_url}/v1/audio/transcriptions"), UPLOAD).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(response.text().await.unwrap(), ERROR);
}

#[tokio::test]
async fn an_upload_without_a_declared_length_is_streamed_upstream() {
    let upstream = Upstream::new(StatusCode::OK, "application/json", TRANSCRIPTION);
    let gateway_url = gateway(&upstream.spawn().await).await;

    let response = upload(format!("{gateway_url}/v1/audio/transcriptions"), chunked_upload()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let received = upstream.received.lock().await;
    assert_eq!(&received.body[..], UPLOAD);
    assert!(
        received.headers.get(http::header::CONTENT_LENGTH).is_none(),
        "an undeclared length must stay undeclared upstream"
    );
}

#[tokio::test]
async fn a_declared_oversized_upload_is_rejected_before_the_upstream_is_called() {
    let upstream = Upstream::new(StatusCode::OK, "application/json", TRANSCRIPTION);
    let gateway_url = gateway_with_limit(&upstream.spawn().await, 64).await;

    let response = upload(format!("{gateway_url}/v1/audio/transcriptions"), UPLOAD).await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_undeclared_oversized_upload_is_cut_off() {
    let upstream = Upstream::new(StatusCode::OK, "application/json", TRANSCRIPTION);
    let gateway_url = gateway_with_limit(&upstream.spawn().await, 64).await;

    let response = upload(format!("{gateway_url}/v1/audio/transcriptions"), chunked_upload()).await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
