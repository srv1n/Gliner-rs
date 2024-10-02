use serde::{Deserialize, Serialize};

pub type RawInferenceResult = Vec<Vec<(String, usize, usize, String, f32)>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityResult {
    pub span_text: String,
    pub start: usize,
    pub end: usize,
    pub label: String,
    pub score: f32,
}

impl std::fmt::Display for EntityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} | {} | (score: {})| ({}-{})",
            self.span_text, self.label, self.score, self.start, self.end
        )
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResultSingle {
    pub entities: Vec<EntityResult>,
}
pub type InferenceResultMultiple = Vec<InferenceResultSingle>;
