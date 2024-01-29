# This is a pipeline for benchmarking scale-invariant generative models.
### 1. Add your custom model as a single python file, containing a single class whose name is the same as the filename.
### 2. Custom model needs to have forward and sample functions defined. Sample takes requires three arguments: num_samples, resolution, device. Forward takes a single argument.
### 3. Include an additional field under the details section in the model configuration file "config.yaml". The name of the additional field has to be the same as the custom model's class name. Under this field, include all arguments for the model's constructor.
### 4. If model is GAN based, modify the "is_gan" field accordingly.
## Training and Testing
### 5. Choose model to be trained using the "name" field under the model section.
### 6. Use save_name under the train section to decide the name of the saved model files.
### 7. Use the shape_setting field to load dataset of varying resolution. Content must be a list of lists, where each inner list has its first element being the resolution and second element being the proportion of samples at that resolution. The proportion are used to parameterize a categorical distribution to draw sampels at different resolutions.
## MMD Test
### 8. Use the MMD_test section to configure MMD test setting and use MMD_test.py to run MMD test.