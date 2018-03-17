%make training set
MakeCellData();

%%%%%%%%%%%%%%%%%
%define CNN layers to be used in RCNN
%%%%%%%%%%%%%%%%%
%chose input size 32x32, since all image are greater than 16x16 at least
inputLayer = imageInputLayer([64 64 3]);

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
    fullyConnectedLayer(2) %num classes + background
    
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
options = trainingOptions('sgdm',...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-6);

%do training
%Set random seed to ensure reproducibility
rng(42);

%train the RCNN object detector
% detector = trainFasterRCNNObjectDetector(trainingData,layers,options,...
%     'NegativeOverlapRange',[0 0.2], ...
%     'PositiveOverlapRange',[0.3 1], ...
%     'BoxPyramidScale',1.2);
detector = trainFasterRCNNObjectDetector(trainingData,layers,options);

 I = imread(testData.imageFilename{1});
 [bboxes,scores] = detect(detector,I)