clc;
clear all;
close all;

% Escolhendo alguns valores iniciais
nValue = 0.1; % Variação máxima do ruído
trainSize = 800;
weightsMax = 100;

% Gerando pontos de 1 a 10 do X do Seno
X = 0.01:0.01:10;
Y = sin(2*X);

% Adicionando Rúido ao seno
noise = nValue*randn(1, length(Y)) - nValue/2;
Y = Y + noise;

% Randomizando e separando entre treino e teste
randomIndex = randperm(length(X));
xTrain = X(randomIndex(1:trainSize));
xTest = X(randomIndex(trainSize+1:length(X)));
yTrain = Y(randomIndex(1:trainSize));
yTest = Y(randomIndex(trainSize+1:length(Y)));


bestMSE = [];
for quantity = 1:weightsMax
    
    % Selecionando pesos
    hiddenW = rand(quantity,1);
    
    % Calculando H e aplicando a sigmoidal
    H = hiddenW * xTrain;
    H = 1 ./ (1 + exp(-H));
    
    % Peso saída
    exitW = yTrain * pinv(H);
    
    % Testando erro quadrático dos pesos encontrados
    H2 = hiddenW * xTest;
    H2 = 1 ./ (1 + exp(-H2));
    
    yFinal = exitW * H2;
    MSE = sqrt(sum((yFinal - yTest).^2));
    bestMSE(quantity) = MSE;
end
[min,pos] = min(bestMSE)
% plot(xTest,yFinal, 'r*'); 
% hold on;
% plot(xTest,yTest, 'b*');
%plot(X,Y, 'b');