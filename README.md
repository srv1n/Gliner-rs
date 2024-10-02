# GlinerRust [WIP]

GlinerRust is a Rust library for Named Entity Recognition (NER) using ONNX models. It provides a simple and efficient way to perform NER tasks on text data using pre-trained models.

## Features

- Easy-to-use API for Named Entity Recognition
- Support for custom ONNX models
- Asynchronous processing for improved performance
- Configurable parameters for fine-tuning

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
glinerrust = { git = "https://github.com/srv1n/Gliner-rs.git" }
```

Alternatively, you can clone the repository and use it locally:

```
git clone https://github.com/srv1n/Gliner-rs.git
cd Gliner-rs
```

Then, in your `Cargo.toml`, add:

```toml
[dependencies]
glinerrust = { path = "path/to/Gliner-rs" }
```

## Running the Example

To run the provided example:

1. Ensure you have Rust and Cargo installed on your system.
2. Clone this repository:
   ```
   git clone https://github.com/yourusername/glinerrust.git
   cd glinerrust
   ```
3. Download the required model and tokenizer files:
   - Place `tokenizer.json` in the project root
   - Place `model_quantized.onnx` in the project root
4. Run the example using Cargo:
   ```
   cargo run --example basic_usage
   ```

Note: Make sure you have the necessary ONNX model and tokenizer files before running the example. The specific model and tokenizer files required depend on your use case and the pre-trained model you're using.

## Configuration

The `InitConfig` struct allows you to customize the behavior of GlinerRust:

- `tokenizer_path`: Path to the tokenizer JSON file
- `model_path`: Path to the ONNX model file
- `max_width`: Maximum width for processing (optional)
- `num_threads`: Number of threads to use for inference (optional)

## API Reference

### `Gliner`

The main struct for interacting with the GlinerRust library.

#### Methods

- `new(config: InitConfig) -> Self`: Create a new Gliner instance
- `initialize(&mut self) -> Result<(), GlinerError>`: Initialize the Gliner instance
- `inference(&self, input_texts: &[String], entities: &[String], ignore_subwords: bool, threshold: f32) -> Result<Vec<InferenceResultSingle>, GlinerError>`: Perform inference on the given input texts

### `InitConfig`

Configuration struct for initializing a Gliner instance.

### `EntityResult`

Represents a single entity detected in the text.

### `InferenceResultSingle`

Represents the inference result for a single input text.

### `InferenceResultMultiple`

Represents the inference results for multiple input texts.

## Error Handling

The library uses a custom `GlinerError` enum for error handling, which includes:

- `InitializationError`: Errors that occur during initialization
- `InferenceError`: Errors that occur during the inference process

## Performance Considerations

- The `num_threads` option in `InitConfig` allows you to control the number of threads used for inference. Adjust this based on your system's capabilities.
- The `max_width` option can be used to limit the maximum input size. This can help manage memory usage for large inputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Support

If you encounter any issues or have questions, please file an issue on the GitHub repository.

