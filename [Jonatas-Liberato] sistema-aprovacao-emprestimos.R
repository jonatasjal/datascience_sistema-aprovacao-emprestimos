#### SISTEMA DE APROVA??O DE EMPR?STIMOS ####

# O objetivo desse projeto ? utilizar ?rovres de Decis?o em um case cl?ssico, que ? a aprova??o de empr?stimos
# Apesar de podermos realizar diversas t?cnicas, faemos o simples visando ?rvores de Decis?o

# === COMO FUNCIONA A APROVA??O

# A primeira etapa ? verificar se h? saldo em conta corrente
# SIM > Cliente aprovado
# N?O > Verifica Aplica??es:
#         SIM > Cliente aprovado
#         N?O > Cliente com risco

#-------

#Aqui definimos quais atributos constituir?o a ordens da ?rvore como n?s, quebras 
# e demais, para isso respondemos as quest?es:
#   1. Qual atributo inicia a ?rovre?
#   2. Qual atributo vem em seguida?
#   3. Quando devemos parar de construir ramos (overfitting)?
  
#-------

#  Usamos um dos seguintes indicadores:
#  - Entropia(desordem dos dados)
#  - ?ndice de Gini
#  - Taxa de Ganho

#-------

# Uma estrat?gia para definir o n? raiz e a dividir o conjunto de 
# dados, ? o Greedy Selection.

# ------

# USO:
#  Em uma divis?o entre 0 e 1:
  
# N?O-HOMOG?NEA (alto grau de impureza)
#   C0: 5 (50%)
#   C1: 5 (50%)

# HOMOG?NEA (baixo grau de impureza)
#   C0: 9 (90%)
#   C1: 1 (10%)

# Ela calcula a impureza do n? (utiliza o n?vel da entropia).

# -------

# L?GICA: a ordem ? da de menor impureza como n?s raiz e vai quebrando de acordo 
# com as vari?veis com menos impurezas

# Usa-se  algoritmos ID3, C4.5 e C5.0 para calcular o n? raiz com base em quanto 
# do total de Entropia ? reduzido se tal n? ? escolhido

# Entropia ? aplicada para computar o ganho de informa??o para todos os atributos. 
# O com maior ganho de informa??o ? escolhido e assim ? testado par cada n? a fim 
# de escolher o menor.

# Ou seja, SEPARA MELHOR OS DADOS

# -------

# DATASET

# https://drive.google.com/file/d/1OOqU0QS1e6NLTscCz85O2y6PjJJeJ7vV/view?usp=sharing


# ================================ IN?CIO ================================

getwd()
setwd("C:/Users/jonat/OneDrive/__ARQUIVOS DE CURSOS/_____PORTFOLIO/R/[Jonatas-Liberato] sistema-aprovacao-emprestimo")

# F?RMULA PARA CALCULAR A ENTROPIA DE 2 CLASSES (TEM QUE DAR 100%)
-0.9999 * log(0.9999) - 0.0001 * log2(0.0001)

# GERANDO CURVA DE ENTROPIA
curve(-x * log2(x) - (1 - x) * log2(1 - x), col = "red", xlab = "x", ylab = "Entropy", lwd = 4)

# -------

# DATASET
dataset <- read.csv('credito.csv')
str(dataset)
View(dataset)

# obs: notamos que algumas vari?veis foram transformadas em factor (qualitativas)

# -------

# ANALISANDO ATRIBUTOS ESPEC?FICOS

# tablet() mostra um resumo dos dados em formato de tabela
table(dataset$checking_balance)
table(dataset$savings_balance)

# -------

# ANALISANDO OS PERCENTIS

summary(dataset$months_loan_duration)
summary(dataset$amount)

# -------

# INVESTIGADO O TARGET (default)

table(dataset$default)
# nota-se um desbalanceamento de classe, o SMOTH seria ideal, mas n?o ser? feito

# -------

# CONSTRUINDO OS DADOS DE TREINO E DE TESTE

set.seed(123)
train_sample <- sample(1000, 900)
# poder?amos usar o train_test_split() mas o objetivo ? mais simples


# -------

# FAZENDO SPLIT DOS DATAFRAMES

dataset_train <- dataset[train_sample, ]
dataset_test <- dataset[-train_sample, ]

# -------

# PROPOR??O

prop.table(table(dataset_train$default))
prop.table(table(dataset_test$default))

# -------

# INSTALANDO O C50

install.packages("C50")
library(C50)
?C5.0

# -------

# CONSTRUINDO O MODELO

# Selecionando as explicativas menos o target (defalut)
dataset_model <- C5.0(dataset_train[-17], dataset_train$default)
dataset_model

# -------

# ANALISANDO OS RESULTADOS DA ?RVORE

summary(dataset_model)

# -------

# AVALIANDO A PERFORMANCE

dataset_pred <- predict(dataset_model, dataset_test)

# -------

# USANDO A MATRIX DE CONFUS?O PARA COMPARAR VALORES OBSERVADOS E PREVISTOS

install.packages('gmodels')
library(gmodels)

# -------

# MATRIX DE CONFUS?O

CrossTable(dataset_test$default,
           dataset_pred,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c('Observado', 'Previsto'))


# -------

# MELHORANDO A PERFORMANCE

dataset_boost <- C5.0(dataset_train[-17], dataset_train$default, trials = 10)
dataset_boost
summary(dataset_boost)


# -------

# SCORE DO MODELO

dataset_boost_pred <- predict(dataset_boost, dataset_test)

# -------

# MATRIX DE CONFUS?O DE BOOST

CrossTable(dataset_test$default,
           dataset_boost_pred,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c('Observado', 'Previsto'))

# -------

# CLASSIFICANDO E DANDO PESO A ERROS

# Matrix de custos
matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("Previsto", "Observado")
matrix_dimensions

# Construindo a Matrix
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost

# -------

# APLICANDO A MATRIX ? ?RVORE

dataset_cost <- C5.0(dataset_train[-17], dataset_train$default, costs = error_cost)

# -------

# VERIFICANDO O SCORE
dataset_cost_pred <- predict(dataset_cost, dataset_test)

# -------

# Confusion Matrix
CrossTable(dataset_test$default, 
           dataset_cost_pred,
           prop.chisq = FALSE, 
           prop.c = FALSE, 
           prop.r = FALSE,
           dnn = c('Observado', 'Previsto'))