load('featuresTrain.mat');
load('featuresValidation.mat');

trainLabels = categorical(trainLabels);
validationLabels = categorical(validationLabels);


svmModel = fitcecoc(featuresTrain, trainLabels);


save('svmModel.mat', 'svmModel');


predictedLabels = predict(svmModel, featuresValidation);


accuracy = mean(predictedLabels == validationLabels);
disp(['Validation accuracy: ', num2str(accuracy)]);


figure;
confusionchart(validationLabels, predictedLabels);
title('Confusion Matrix for Validation Data');
