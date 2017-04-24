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

% Criando folds

xyTrain = [xTrain ; yTrain];
sizeFold = length(xTrain)/5;

for i = 1 : 5
    fold(i,:,:) = xyTrain(:, sizeFold*(i-1)+1 : sizeFold*i);
end

%Treino

% Aplicando ELM a fold específico

centerMSE = [];
for quantity = 1:weightsMax
    
    mediaFold = 0;
    for removedFold = 1 : 5
        
        %Usando folds escolhidos para treino
        xTestFold = [];
        yTestFold = [];
        yFold = [];
        xFold = [];
        
        for i = 1 : 5
            if i ~= removedFold
                for j = 1 : length(fold)
                    xFold = [xFold fold(i,1,j)];
                    yFold = [yFold fold(i,2,j)];
                end
            else
                for j = 1 : length(fold)
                    xTestFold = [xTestFold fold(i,1,j)];
                    yTestFold = [yTestFold fold(i,2,j)];
                end
            end
        end
        
        
        
        % Selecionando pesos
        hiddenW = rand(quantity,1);
        
        % Calculando H e aplicando a sigmoidal
        H = hiddenW * xFold;
        H = 1 ./ (1 + exp(-H));
        
        % Peso saída
        exitW = yFold * pinv(H);
        
        % Testando erro quadrático dos pesos encontrados
        H2 = hiddenW * xTestFold;
        H2 = 1 ./ (1 + exp(-H2));
        
        yFinal = exitW * H2;
        MSE = sqrt(sum((yFinal - yTestFold).^2));
        mediaFold = mediaFold+MSE;
        
    end
    centerMSE(quantity) = mediaFold/5;
end
[min,centerQTD] = min(centerMSE);

% TESTE

% Selecionando pesos
hiddenWTest = rand(centerQTD,1);

% Calculando H e aplicando a sigmoidal
Htest = hiddenWTest * xTrain;
Htest = 1 ./ (1 + exp(-Htest));

% Peso saída
exitWTest = yTrain * pinv(Htest);

% Testando erro quadrático dos pesos encontrados
H2test = hiddenWTest * xTest;
H2test = 1 ./ (1 + exp(-H2test));

yFinalTest = exitWTest * H2test;
MSE = sqrt(sum((yFinalTest - yTest).^2));

plot(xTest,yFinalTest, 'r*'); 
hold on;
plot(xTest,yTest, 'b.');