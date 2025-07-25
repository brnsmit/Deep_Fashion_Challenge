# Deep Fashion Challenge - Vision Transformer Classification

Este projeto implementa um classificador de moda usando Vision Transformer (ViT) para categorizar diferentes tipos de roupas e acessórios. O modelo é treinado em um subset do dataset de moda e utiliza técnicas modernas de deep learning para classificação de imagens.

## 1. Objetivo

Desenvolver um sistema de classificação automática de itens de moda utilizando Vision Transformers (ViT) pré-treinados, demonstrando a eficácia dessa arquitetura em tarefas de classificação de imagens de moda.

## 2. Características

- **Modelo**: Vision Transformer (ViT-B/16) pré-treinado
- **Classes**: 5 categorias de moda selecionadas aleatoriamente
- **Dataset**: 50 imagens por classe (250 imagens total)
- **Divisão**: 80% treino / 20% teste
- **Avaliação**: Métricas completas + matriz de confusão
- **Visualização**: Curvas de loss e acurácia durante o treinamento

## 3. Pré-requisitos

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy
pandas
pillow
matplotlib
seaborn
scikit-learn
tqdm
```

## 4. Instalação

4.1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/deep-fashion-challenge.git
cd deep-fashion-challenge
```

4.2. Instale as dependências:
```bash
pip install torch torchvision numpy pandas pillow matplotlib seaborn scikit-learn tqdm
```

4.3. Configure o dataset:
   - Organize suas imagens de moda em pastas por categoria
   - Ajuste o caminho `root_dir` no código para apontar para seu dataset

## 5. Estrutura do Dataset

```
dataset/
├── img/
│   ├── categoria_1/
│   │   ├── imagem1.jpg
│   │   ├── imagem2.jpg
│   │   └── ...
│   ├── categoria_2/
│   │   ├── imagem1.jpg
│   │   └── ...
│   └── ...
```

## 6. Como Usar

### Treinamento

```python
# O código automaticamente:
# 1. Seleciona 5 classes aleatórias com pelo menos 50 imagens
# 2. Carrega o dataset com transformações apropriadas
# 3. Inicializa o modelo ViT pré-treinado
# 4. Treina por 5 épocas
# 5. Exibe curvas de treinamento

python deep_fashion_challenge.py
```

### Principais Componentes

#### 6.1. **Carregamento de Dados**
- Seleção inteligente de classes com amostras suficientes
- Dataset customizado com balanceamento automático
- Transformações padronizadas para ViT (224x224, normalização)

#### 6.2. **Modelo ViT**
- Vision Transformer Base (ViT-B/16) pré-treinado
- Cabeça de classificação customizada para 5 classes
- Otimizador Adam com learning rate adaptado

#### 6.3. **Treinamento**
- Monitoramento de loss e acurácia por época
- Visualização em tempo real das métricas
- Uso eficiente de GPU quando disponível

#### 6.4. **Avaliação**
- Relatório completo de classificação
- Matriz de confusão com visualização
- Métricas por classe (precision, recall, f1-score)

## 7. Resultados

O modelo gera automaticamente:

- **Curvas de Treinamento**: Evolução do loss e acurácia
- **Relatório de Classificação**: Métricas detalhadas por classe
- **Matriz de Confusão**: Visualização dos acertos e erros
- **Acurácia Final**: Performance no conjunto de teste

## 8. Personalização

### Alterar Número de Classes
```python
# Modifique esta linha para selecionar mais/menos classes
selected_classes = random.sample(valid_classes, 5)  # Altere o 5
```

### Ajustar Hiperparâmetros
```python
# Batch size
train_loader = DataLoader(train_dataset, batch_size=16, ...)  # Altere batch_size

# Learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Altere lr

# Número de épocas
train_model(model, train_loader, epochs=5)  # Altere epochs
```

### Amostras por Classe
```python
# Modifique samples_per_class no dataset
dataset = SmallFashionDataset(root_dir, selected_classes, 
                             transform=transform, samples_per_class=50)
```

## 9. Arquitetura do Projeto

```
deep-fashion-challenge/
├── deep_fashion_challenge.py    # Código principal
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências
└── results/                     # Resultados e visualizações
    ├── training_curves.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## 10. Melhorias Futuras

- [ ] Implementar data augmentation avançado
- [ ] Adicionar validação cruzada
- [ ] Suporte para mais arquiteturas (ResNet, EfficientNet)
- [ ] Interface web para classificação em tempo real
- [ ] Otimização de hiperparâmetros automática
- [ ] Métricas de interpretabilidade do modelo

## Autor

- **Bruno Paiva Smit de Freitas** - *Trabalho inicial* - [brnsmit](https://github.com/brnsmit)
