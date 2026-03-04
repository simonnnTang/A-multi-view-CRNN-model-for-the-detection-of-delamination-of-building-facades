# A-multi-view-CRNN-model-for-the-detection-of-delamination-of-building-facades
A local CRNN model for hollow/non-hollow classification.
The code execution order is audio, spec, b1, b2, fusion. During runtime, the code files and your data should be placed in the same folder.

The audio preprocessor the audio data and serves as the input to the b1 model. 
The spec_preprocessor extracts three different spectrograms of the audio and integrates them into a three-channel image, which is the input to the b2 model. 
b1 and b2 are two independent branches of the model, used for ablation experiments.

The fusion module, as the final CRNN model, uses a GRU to fuse the features input from the two branches for final classification.
