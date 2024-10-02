use crate::types::RawInferenceResult;
use std::collections::HashMap;

// Helper functions
fn is_nested(idx1: &[usize], idx2: &[usize]) -> bool {
    (idx1[0] <= idx2[0] && idx1[1] >= idx2[1]) || (idx2[0] <= idx1[0] && idx2[1] >= idx1[1])
}

fn has_overlapping(idx1: &[usize], idx2: &[usize], multi_label: bool) -> bool {
    if idx1[0..2] == idx2[0..2] {
        return !multi_label;
    }
    !(idx1[0] > idx2[1] || idx2[0] > idx1[1])
}

fn has_overlapping_nested(idx1: &[usize], idx2: &[usize], multi_label: bool) -> bool {
    if idx1[0..2] == idx2[0..2] {
        return !multi_label;
    }
    !(idx1[0] > idx2[1] || idx2[0] > idx1[1] || is_nested(idx1, idx2))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub trait Decoder {
    fn decode(
        &self,
        batch_size: usize,
        input_length: usize,
        max_width: usize,
        num_entities: usize,
        texts: &[String],
        batch_ids: &[usize],
        batch_words_start_idx: &[Vec<usize>],
        batch_words_end_idx: &[Vec<usize>],
        id_to_class: &HashMap<usize, String>,
        model_output: &[f32],
        flat_ner: bool,
        threshold: f32,
        multi_label: bool,
    ) -> RawInferenceResult;
}

// BaseDecoder trait
pub trait BaseDecoder {
    fn new(config: HashMap<String, String>) -> Self
    where
        Self: Sized;
    fn decode(
        &self,
        batch_size: usize,
        input_length: usize,
        max_width: usize,
        num_entities: usize,
        texts: &[Vec<String>],
        batch_ids: &[usize],
        batch_words_start_idx: &[Vec<usize>],
        batch_words_end_idx: &[Vec<usize>],
        id_to_class: &HashMap<usize, String>,
        model_output: &[f32],
        flat_ner: bool,
        threshold: f32,
        multi_label: bool,
    ) -> RawInferenceResult;

    fn greedy_search(
        &self,
        spans: &mut Vec<Vec<f32>>,
        flat_ner: bool,
        multi_label: bool,
    ) -> Vec<Vec<f32>> {
        let has_ov = if flat_ner {
            Box::new(move |idx1: &[usize], idx2: &[usize]| has_overlapping(idx1, idx2, multi_label))
                as Box<dyn Fn(&[usize], &[usize]) -> bool>
        } else {
            Box::new(move |idx1: &[usize], idx2: &[usize]| {
                has_overlapping_nested(idx1, idx2, multi_label)
            }) as Box<dyn Fn(&[usize], &[usize]) -> bool>
        };

        let mut new_list: Vec<Vec<f32>> = Vec::new();
        spans.sort_by(|a, b| b.last().unwrap().partial_cmp(a.last().unwrap()).unwrap());

        for b in spans.iter() {
            let mut flag = false;
            for new_span in &new_list {
                if has_ov(
                    &b[0..2].iter().map(|&x| x as usize).collect::<Vec<usize>>(),
                    &new_span[0..2]
                        .iter()
                        .map(|&x| x as usize)
                        .collect::<Vec<usize>>(),
                ) {
                    flag = true;
                    break;
                }
            }
            if !flag {
                new_list.push(b.clone());
            }
        }

        new_list.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        new_list
    }
}

// SpanDecoder struct
pub struct SpanDecoder {
    config: HashMap<String, String>,
}

impl BaseDecoder for SpanDecoder {
    fn new(config: HashMap<String, String>) -> Self {
        SpanDecoder { config }
    }

    fn decode(
        &self,
        batch_size: usize,
        input_length: usize,
        max_width: usize,
        num_entities: usize,
        texts: &[Vec<String>],
        batch_ids: &[usize],
        batch_words_start_idx: &[Vec<usize>],
        batch_words_end_idx: &[Vec<usize>],
        id_to_class: &HashMap<usize, String>,
        model_output: &[f32],
        flat_ner: bool,
        threshold: f32,
        multi_label: bool,
    ) -> RawInferenceResult {
        let mut spans: Vec<Vec<Vec<f32>>> = vec![vec![]; batch_size];

        let batch_padding = input_length * max_width * num_entities;
        // println!("batch_padding {:?}", batch_padding);
        let start_token_padding = max_width * num_entities;
        let end_token_padding = num_entities;

        for (id, &value) in model_output.iter().enumerate() {
            let batch = id / batch_padding;
            let start_token = (id / start_token_padding) % input_length;
            let end_token = start_token + ((id / end_token_padding) % max_width);

            let entity = id % num_entities;

            let prob = sigmoid(value);

            if prob >= threshold
                && start_token < batch_words_start_idx[batch].len()
                && end_token < batch_words_end_idx[batch].len()
            {
                let global_batch = batch_ids[batch];
                let start_idx = batch_words_start_idx[batch][start_token];
                let end_idx = batch_words_end_idx[batch][end_token];

                // println!("entity: {}", id_to_class.get(&entity).unwrap());
                // let span_text = texts[global_batch][0][start_idx..end_idx].to_string();

                // println!("span_text: {}", span_text);

                spans[batch].push(vec![
                    start_idx as f32,
                    end_idx as f32,
                    entity as f32, // Changed from (entity + 1) to entity
                    prob,
                ]);
            }
        }

        let mut all_selected_spans: RawInferenceResult = Vec::new();

        for (id, res_i) in spans.iter_mut().enumerate() {
            let selected_spans = self.greedy_search(res_i, flat_ner, multi_label);
            let batch_spans: Vec<(String, usize, usize, String, f32)> = selected_spans
                .into_iter()
                .map(|span| {
                    let start_idx = span[0] as usize;
                    let end_idx = span[1] as usize;
                    let entity_id = span[2] as usize;
                    let prob = span[3];
                    let global_batch = batch_ids[id];

                    // Ensure indices are within bounds
                    let span_text = if start_idx < texts[global_batch][0].len()
                        && end_idx <= texts[global_batch][0].len()
                    {
                        texts[global_batch][0][start_idx..end_idx].to_string()
                    } else {
                        // println!("Issue with start_idx or end_idx");
                        String::new() // or handle the error as needed
                    };

                    let entity_label = id_to_class
                        .get(&entity_id)
                        .unwrap_or(&"UNKNOWN".to_string())
                        .clone();
                    (span_text, start_idx, end_idx, entity_label, prob)
                })
                .collect();

            all_selected_spans.push(batch_spans);
        }

        all_selected_spans
    }
}

impl Decoder for SpanDecoder {
    fn decode(
        &self,
        batch_size: usize,
        input_length: usize,
        max_width: usize,
        num_entities: usize,
        texts: &[String],
        batch_ids: &[usize],
        batch_words_start_idx: &[Vec<usize>],
        batch_words_end_idx: &[Vec<usize>],
        id_to_class: &HashMap<usize, String>,
        model_output: &[f32],
        flat_ner: bool,
        threshold: f32,
        multi_label: bool,
    ) -> RawInferenceResult {
        // Convert texts to Vec<Vec<String>>
        let texts_vec: Vec<Vec<String>> = texts.iter().map(|s| vec![s.clone()]).collect();

        // Call the BaseDecoder implementation
        BaseDecoder::decode(
            self,
            batch_size,
            input_length,
            max_width,
            num_entities,
            &texts_vec,
            batch_ids,
            batch_words_start_idx,
            batch_words_end_idx,
            id_to_class,
            model_output,
            flat_ner,
            threshold,
            multi_label,
        )
    }
}
