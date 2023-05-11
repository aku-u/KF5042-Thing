clc; clear; % clears the screen and workspace
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt')); % This will open the positive lexicon text file.
C = textscan(fidPositive,'%s','CommentStyle',';'); %skips commented lines
wordsPositive = string(C{1});
fclose all; %Closes all Files

words_hash = java.util.Hashtable;
[possize, ~] = size(wordsPositive); 
for ii = 1:possize % While ii = 1
words_hash.put(wordsPositive(ii,1),1); %Put the words in the positive wordhash
end
%**************************************
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';'); %Same process as positive  
wordsNegative = string(C{1});
fclose all;

[negsize, ~] = size(wordsNegative); 
for ii = 1:negsize
words_hash.put(wordsNegative(ii,1),-1);%Put the words in the negative wordhash
end

filename = "AmazonDataSet.txt"; %The file that is being opened
dataReviews = readtable(filename,'TextType','string');  
textData = dataReviews.review; %gets the review text 
actualScore = dataReviews.score; %gets the review sentiment

sents = preprocessReviews(textData);
fprintf('File: %s, Sentences: %d\n', filename, size(sents));

sentimentScore = zeros(size(sents));
for ii = 1 : sents.length %Loops the length of sents.length
    docwords = sents(ii).Vocabulary;
    for jj = 1 : length(docwords)
        if words_hash.containsKey(docwords(jj))
            sentimentScore(ii) = sentimentScore(ii) + words_hash.get(docwords(jj));
        end
    end
    fprintf('Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ii, joinWords(sents(ii)), sentimentScore(ii), actualScore(ii)); %Prints the sentiment scores, sentences and actual score.
end

sentimentScore(sentimentScore > 0) = 1;   %take anything over 1 to be 1 only
sentimentScore(sentimentScore < 0)= -1;   %take anything under 0 to be -1, so no neutrals only negatives.

notfound = sum(sentimentScore == 0); %Not found is all of the sentiment scores that = 0
covered = numel(sentimentScore)- notfound; %Covered is the number of sentiment scores - not found
tp=0; tn=0; count=0;
for i=1:length(actualScore) %Loops the amount of actual scores found
    if sentimentScore(i)==1 && actualScore(i)==1
        tp=tp+1; count=count+1; %If the score is positive (1), adds 1 to the TP counter
    elseif sentimentScore(i)==-1 && actualScore(i)==0
        tn=tn+1; count=count+1; %If not, adds to the negative counter.
    end
end
accuracy = (tp+tn)*100/covered 
coverage=covered*100/numel(sentimentScore) %The Accuracy and Coverage Calculations

load wordembedding
words = [wordsPositive;wordsNegative]; 
labels = categorical(nan(numel(words),1)); 
labels(1:numel(wordsPositive)) = "Positive";
labels(numel(wordsPositive)+1:end) = "Negative"; 

data = table(words,labels,'VariableNames',{'Word','Label'});
idx=~isVocabularyWord(emb,data.Word);
data(idx,:) = [];

numWords = size(data,1); 
cvp = cvpartition(numWords,'HoldOut',0.01); %holdout fewer if applying model
dataTrain = data(training(cvp),:); 
dataTest = data(test(cvp),:);
%Convert the words in the training data to word vectors using word2vec. 
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

%Train a support vector machine (SVM) Sentiment Classifier which classifies word vectors into positive and negative categories.

model = fitcsvm(XTrain,YTrain);

%Test Classifier

wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest); 
YTest = dataTest.Label;

%Predicts the sentiment labels of the test word vectors. 
[YPred,scores] = predict(model,XTest);

%Visualizes the classification in a confusion matrix. 
figure
confusionchart(YTest,YPred, 'ColumnSummary','column-normalized'); 

idx = ~isVocabularyWord(emb,sents.Vocabulary);
removeWords(sents, idx); %Strips words

sentimentScore = zeros(size(sents)); 
notFound=0; %Sets the Sentiment score to the size of sents, not found to 0
for ii = 1 : sents.length
    docwords = sents(ii).Vocabulary;
    for jj = 1 : length(docwords)
        vec = word2vec(emb,docwords); %Sets the vector to be all words set to vector
        if any(any(isnan(vec)))
            notFound=notFound+1;
            sentimentScore(ii) = 0;
        else
            [~,scores] = predict(model,vec);
            sentimentScore(ii) = mean(scores(:,1));
        end
    end
    fprintf('Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ii, joinWords(sents(ii)), sentimentScore(ii), actualScore(ii));
end

sentimentScore(sentimentScore > 0) = 1;  
sentimentScore(sentimentScore < 0)= -1;   %Same Method as Prior


notfound = sum(sentimentScore == 0);
covered = numel(sentimentScore)- notfound;

tp=0; tn=0; count=0;
for i=1:length(actualScore) %Same method as prior in code
    if sentimentScore(i)==1 && actualScore(i)==1
        tp=tp+1; count=count+1;
    elseif sentimentScore(i)==-1 && actualScore(i)==0
        tn=tn+1; count=count+1;
    end
end
accuracy = (tp+tn)*100/covered 
coverage=covered*100/numel(sentimentScore)


