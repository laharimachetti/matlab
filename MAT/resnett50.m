net = resnet50;

lgraph = layerGraph(net);

layersToRemove = {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'};
lgraph = removeLayers(lgraph, layersToRemove);

emotionData = imageDatastore('C:\Users\user\Desktop\MATLAB\MAT\CS229-master (1)\CS229-master\TFEID Slight', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[emotionTrain, emotionValidation] = splitEachLabel(emotionData, 0.8, 'randomize');

augmenter = imageDataAugmenter('RandRotation', [-10, 10], 'RandScale', [0.8 1.2]);
augmentedTrain = augmentedImageDatastore([224 224], emotionTrain, 'DataAugmentation', augmenter);
augmentedValidation = augmentedImageDatastore([224 224], emotionValidation);


numClasses = numel(categories(emotionTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc8', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    flattenLayer('Name','flatten')
    classificationLayer('Name', 'classification')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc8');

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ... 
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedValidation, ...
    'ValidationFrequency', floor(numel(augmentedTrain.Files) / 32), ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'L2Regularization', 1e-4, ...
    'GradientThresholdMethod', 'l2norm', ...
    'GradientThreshold', 1.0, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10);

featureNet = trainNetwork(augmentedTrain, lgraph, options);

featuresTrain = activations(featureNet, augmentedTrain, 'avg_pool', 'OutputAs', 'rows');
featuresValidation = activations(featureNet, augmentedValidation, 'avg_pool', 'OutputAs', 'rows');


trainLabels = emotionTrain.Labels;
validationLabels = emotionValidation.Labels;
save('featuresTrain.mat', 'featuresTrain', 'trainLabels');
save('featuresValidation.mat', 'featuresValidation', 'validationLabels');


save('featureNet.mat', 'featureNet');
