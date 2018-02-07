%load all data
data = load('fasterRCNNVehicleTrainingData.mat');
vehicleDataset = data.vehicleTrainingData;

%sample the data
vehicleDataset(1:4,:) %array with 2 columns - imageFilename, vehicle

%images are actually located in a separate directory
dataDir = fullfile(toolboxdir('vision'),'visiondata');

%modify the filename column to contain the complete directory path
vehicleDataset.imageFilename = fullfile(dataDir,vehicleDataset.imageFilename);

%%read in a sample image
%I = imread(vehicleDataset.imageFilename{10});
%%add the rectangles that represents the regions of interest
%I = insertShape(I,'Rectangle',vehicleDataset.vehicle{10});
%I=imresize(I,3);
%figure
%imshow(I)

%make training set
idx = floor(0.6*height(vehicleDataset));
trainingData = vehicleDataset(1:idx,:);
testData = vehicleDataset(idx:end,:);

%%%%%%%%%%%%%%%%%
%define CNN layers to be used in RCNN
%%%%%%%%%%%%%%%%%
%chose input size 32x32, since all image are greater than 16x16 at least
inputLayer = imageInputLayer([32 32 3]);

%CNN parameters
filterSize = [3 3];
numFilters = 32;

%create the middle layers of the network (conv, relu, max pool)
middleLayers = [
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(3,'Stride',2)
    ];

%final layer of the network (dense, softmax loss)
finalLayers = [
    fullyConnectedLayer(64) %64 filters
    reluLayer()
    fullyConnectedLayer(width(vehicleDataset)) %num classes + background?
    
    %softmax loss
    softmaxLayer()
    classificationLayer()
    ];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

%RCNN has four parts
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%can load a pretrained model
%change this value to true if training a new model
doTrainingAndEval = false;

if doTrainingAndEval
    %Set random seed to ensure reproducibility
    rng(42);
    
    %train the RCNN object detector
    detector = trainFasterRCNNObjectDetector(trainingData,layers,options,...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1], ...
        'BoxPyramidScale',1.2);
else
    detector = data.detector;
end
    
%verify the training
I = imread('highway.png');

%run the trained detector on the loaded image
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

if doTrainingAndEval
    %run the trained detector on each test image
    resultsStruct = struct([]);
    for i = 1:height(testData)
        I = imread(testData.imageFilename{i});
        [bboxes,scores,labels] = detec(detector,I);
        
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;
    end
    
    results = struct2table(resultsStruct);
else
    results = data.results;
end

%get the ground truch values for the evaluated objects
expectedResults = testData(:,2:end);

%evaluate the detector
[ap,recall,precision] = evaluateDetectionPrecision(results,expectedResults);

% Plot precision/recall curve
figure
plot(recall, precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.1f', ap))