# Fundamentos de Visao Computacional

Mestrado em Ciência da Computação - UFSCar

## Projeto 1 - Erros de classificação e test time augmentation

1. Treinar uma resnet 18 para classificar o dataset Oxford Pets, ou utilizar o checkpoint da disciplina;
2. Identificar as 10 imagens de cachorro do conjunto de validação para as quais o modelo "mais errou", ou seja, o modelo associou a menor probabilidade de ser cachorro. Fazer o mesmo para as imagens da classe gato;
3. Realizar a técnica "test time augmentation" e verificar se as probabilidades dessas 10 imagens melhoram. 

A técnica "test time augmentation" consiste em criar diversas versões da mesma imagem usando data augmentation, aplicar o modelo nas imagens aumentadas e calcular a média das probabilidades nas imagens

![](https://github.com/BrunoShimura/fundamentos-de-visao-computacional/blob/main/projeto%201/Images/augmentation.png?raw=true)

## Projeto 2 - Variação de hiperparâmetros do decodificador

1. Para a arquitetura EncoderDecoder implementada em aula, treinar a rede no dataset Oxford Pets variando as configurações do decodificador de acordo com as instruções abaixo
2. Na aula, o decodificador foi construído a partir de 5 tensores de atributos extraídos da ResNet. Esses tensores possuem stride (passo) 2, 4, 8, 16 e 32. Lembrando que o stride define a redução de resolução a partir da imagem original.
3. Modifique o decodificador e treine o modelo EncoderDecoder para verificar a qualidade do resultado. Utilize as seguintes variações do decodificador:
    * Utilizar apenas os atributos no stride 32
    * Utilizar apenas os atributos no stride 2
    * Utilizar os atributos nos strides 2 e 32
    * Utilizar os atributos nos strides 2, 8 e 32
    * Utilizar os atributos nos strides 2 e 32 mas adicionar uma sequência de camadas conv-batchnorm-relu extra para cada bloco do decodificador. 
4. Avalie a qualidade da segmentação em todos os testes em relação ao custo computacional de cada modelo

Lembre-se de utilizar um codificador pré-treinado, o que reduz muito o número de épocas de treinamento.