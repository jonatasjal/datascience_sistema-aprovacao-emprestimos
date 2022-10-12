#### SISTEMA DE APROVAÇÃO DE EMPRÉSTIMOS ####

# O objetivo desse projeto é utilizar Árovres de Decisão em um case clássico, que é a aprovação de empréstimos
# Apesar de podermos realizar diversas técnicas, faemos o simples visando Árvores de Decisão

# === COMO FUNCIONA A APROVAÇÃO

# A primeira etapa é verificar se há saldo em conta corrente
# SIM > Cliente aprovado
# NÃO > Verifica Aplicações:
#         SIM > Cliente aprovado
#         NÃO > Cliente com risco

#-------

#Aqui definimos quais atributos constituirão a ordens da árvore como nós, quebras 
# e demais, para isso respondemos as questões:
#   1. Qual atributo inicia a árovre?
#   2. Qual atributo vem em seguida?
#   3. Quando devemos parar de construir ramos (overfitting)?
  
#-------

#  Usamos um dos seguintes indicadores:
#  - Entropia(desordem dos dados)
#  - Índice de Gini
#  - Taxa de Ganho

#-------

# Uma estratégia para definir o nó raiz e a dividir o conjunto de 
# dados, é o Greedy Selection.

# ------

# USO:
#  Em uma divisão entre 0 e 1:
  
# NÃO-HOMOGÊNEA (alto grau de impureza)
#   C0: 5 (50%)
#   C1: 5 (50%)

# HOMOGÊNEA (baixo grau de impureza)
#   C0: 9 (90%)
#   C1: 1 (10%)

# Ela calcula a impureza do nó (utiliza o nível da entropia).

# -------

# LÓGICA: a ordem é da de menor impureza como nós raiz e vai quebrando de acordo 
# com as variáveis com menos impurezas

# Usa-se  algoritmos ID3, C4.5 e C5.0 para calcular o nó raiz com base em quanto 
# do total de Entropia é reduzido se tal nó é escolhido

# Entropia é aplicada para computar o ganho de informação para todos os atributos. 
# O com maior ganho de informação é escolhido e assim é testado par cada nó a fim 
# de escolher o menor.

# Ou seja, SEPARA MELHOR OS DADOS

# -------

# DATASET

# https://drive.google.com/file/d/1OOqU0QS1e6NLTscCz85O2y6PjJJeJ7vV/view?usp=sharing


# ================================ INÍCIO ================================

getwd()
setwd("C:/Users/jonat/OneDrive/__ARQUIVOS DE CURSOS/_____PORTFOLIO/R/[Jonatas-Liberato] sistema-aprovacao-emprestimo")

# FÓRMULA PARA CALCULAR A ENTROPIA DE 2 CLASSES (TEM QUE DAR 100%)
-0.9999 * log(0.9999) - 0.0001 * log2(0.0001)

# GERANDO CURVA DE ENTROPIA
curve(-x * log2(x) - (1 - x) * log2(1 - x), col = "red", xlab = "x", ylab = "Entropy", lwd = 4)

# -------

# DATASET
dataset <- read.csv('credito.csv')
str(dataset)
View(dataset)

# obs: notamos que algumas variáveis foram transformadas em factor (qualitativas)

# -------

# ANALISANDO ATRIBUTOS ESPECÍFICOS

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
# nota-se um desbalanceamento de classe, o SMOTH seria ideal, mas não será feito

# -------

# CONSTRUINDO OS DADOS DE TREINO E DE TESTE

set.seed(123)
train_sample <- sample(1000, 900)
# poderíamos usar o train_test_split() mas o objetivo é mais simples


# -------

# FAZENDO SPLIT DOS DATAFRAMES

dataset_train <- dataset[train_sample, ]
dataset_test <- dataset[-train_sample, ]

# -------

# PROPORÇÃO

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

# ANALISANDO OS RESULTADOS DA ÁRVORE

summary(dataset_model)

# -------

# AVALIANDO A PERFORMANCE

dataset_pred <- predict(dataset_model, dataset_test)

# -------

# USANDO A MATRIX DE CONFUSÃO PARA COMPARAR VALORES OBSERVADOS E PREVISTOS

install.packages('gmodels')
library(gmodels)

# -------

# MATRIX DE CONFUSÃO

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

# MATRIX DE CONFUSÃO DE BOOST

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

# APLICANDO A MATRIX À ÁRVORE

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