# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Victor Felipe Alves Pinto

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` foi organizada em três blocos
convolucionais seguidos de uma camada de decisão:

- Bloco 1: Conv2D (32 filtros, kernel 3x3) + BatchNormalization + MaxPooling2D
- Bloco 2: Conv2D (64 filtros, kernel 3x3) + BatchNormalization + MaxPooling2D
- Bloco 3: Conv2D (64 filtros, kernel 3x3) + BatchNormalization

Os dois primeiros blocos reduzem a dimensão espacial da imagem (28x28 → 14x14
→ 7x7). O terceiro bloco dispensa o pooling, pois a imagem já se encontra em
resolução reduzida.

Após os blocos convolucionais, a saída passa por Flatten, Dropout (0.5) e uma
camada Dense com 10 neurônios e ativação softmax, correspondente às dez classes
de dígitos.

Estratégia de validação: Foi utilizou a `validation_split=0.1`, reservando 10%
dos dados de treino exclusivamente para validação. O treinamento foi controlado
por EarlyStopping monitorando a perda de validação, com `patience=3` e
`restore_best_weights=True`, garantindo que o modelo final corresponda ao melhor
ponto observado durante o treino, e não simplesmente ao último.

Total de parâmetros: 87.754.

### 2️⃣ Bibliotecas Utilizadas

- TensorFlow / Keras. para a construção, treinamento e conversão do modelo
- Python 3.11 — ambiente de execução

### 3️⃣ Técnica de Otimização do Modelo

Foi aplicada "Dynamic Range Quantization" por meio do
`tf.lite.TFLiteConverter`, com `converter.optimizations = [tf.lite.Optimize.DEFAULT]`.

A técnica visa reduzir a precisão numérica dos pesos do modelo, diminuindo
de forma significativa o tamanho do arquivo com impacto mínimo na acurácia. Esse tipo de otimização é voltada a dispositivos de borda (Edge AI), onde memória e
capacidade de processamento são limitadas.

### 4️⃣ Resultados Obtidos

-  Acurácia final no conjunto de teste: 98,90%
- Tamanho do `model.h5`:1.113.864 bytes (aproximadamente 1,1 MB)
- Tamanho do `model.tflite`: 97.936 bytes (aproximadamente 98 KB)
- Redução obtida: o modelo otimizado ficou aproximadamente 11 vezes menor

O treinamento foi interrompido automaticamente pelo EarlyStopping na sétima
época, tendo o melhor resultado de validação sido registrado na quarta época
(val_loss 0.0307 / val_accuracy 99,18%).

### 5️⃣ Comentários Adicionais (Opcional)

Decisões técnicas:
- Normalização dos pixels para o intervalo [0, 1], dividindo por 255, conforme
  exigido pela especificação do projeto.
- Dropout de 0.5 antes da camada de saída como estratégia de regularização,
  reduzindo o risco de overfitting.
- BatchNormalization após cada camada convolucional, estabilizando e acelerando
  o treinamento.
- Aumento progressivo de filtros (32 → 64) ao longo dos blocos, acompanhando a
  redução da resolução espacial.

Dificuldades encontradas:
Durante a execução, os artefatos gerados (`model.h5` e `model.tflite`) foram
inicialmente salvos na raiz do repositório, e não na pasta do projeto. A causa
foi o uso de caminhos relativos nos scripts, que resolvem a partir do diretório
de onde o comando é executado, e não do local do arquivo `.py`. Os arquivos
foram movidos para o diretório correto antes do commit.

Aprendizados:
O projeto permitiu compreender na prática o fluxo completo de um pipeline de
visão computacional: preparação dos dados, construção da arquitetura,
treinamento com validação, otimização para borda e verificação por inferência.
Em especial, ficou clara a distinção entre conjunto de validação — consultado
repetidamente durante o treino e portanto capaz de influenciar decisões — e
conjunto de teste, utilizado uma única vez ao final e por isso a métrica mais
confiável de generalização.

### 6️⃣ Exemplo de Inferência
Saída do terminal ao executar `run_inference.py` com o modelo otimizado:

Rodando inferencia em 5 amostras usando model.tflite:
Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Comentário: todas as cinco amostras testadas foram classificadas
corretamente. O resultado confirma que a quantização preservou a capacidade
preditiva do modelo, apesar da redução de aproximadamente 11 vezes no tamanho
do arquivo — objetivo central da etapa de otimização para Edge AI.