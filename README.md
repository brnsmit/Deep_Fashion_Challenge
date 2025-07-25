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

#### 1. **Carregamento de Dados**
- Seleção inteligente de classes com amostras suficientes
- Dataset customizado com balanceamento automático
- Transformações padronizadas para ViT (224x224, normalização)

#### 2. **Modelo ViT**
- Vision Transformer Base (ViT-B/16) pré-treinado
- Cabeça de classificação customizada para 5 classes
- Otimizador Adam com learning rate adaptado

#### 3. **Treinamento**
- Monitoramento de loss e acurácia por época
- Visualização em tempo real das métricas
- Uso eficiente de GPU quando disponível

#### 4. **Avaliação**
- Relatório completo de classificação
- Matriz de confusão com visualização
- Métricas por classe (precision, recall, f1-score)

## 7. Resultados Obtidos

### Métricas de Performance

O modelo ViT-B/16 demonstrou sólida performance no dataset de moda:

- **Acurácia Final**: 84% no conjunto de teste
- **Macro Average**: F1-score de 0.83
- **Weighted Average**: F1-score de 0.84
- **Convergência**: Rápida convergência em 5 épocas
- **Loss Final**: ~0.05 (muito baixo, indicando boa generalização)

### Relatório de Classificação Detalhado

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| **Diamond_Print_Halter_Cami_Dress** | 0.87 | 0.93 | 0.90 | 14 |
| **Chic_Sweater_Crop_Top** | 1.00 | 0.78 | 0.88 | 9 |
| **Cutout_A-Line_Dress** | 0.70 | 0.88 | 0.78 | 8 |
| **Contrast-Paneled_Henley** | 0.78 | 0.88 | 0.82 | 8 |
| **Dip-Dyed_Knotted_Sweater** | 0.89 | 0.73 | 0.80 | 11 |
| **Accuracy** | - | - | **0.84** | 50 |
| **Macro Avg** | 0.85 | 0.84 | 0.83 | 50 |
| **Weighted Avg** | 0.85 | 0.84 | 0.84 | 50 |

### Análise das Curvas de Treinamento

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/f0a5b863-8cda-4bdd-8321-5a0cc5fdd948" />


**Curva de Loss:**
- Decréscimo exponencial de ~1.15 para ~0.05
- Convergência estável sem oscilações significativas
- Sem sinais de overfitting

**Curva de Acurácia:**
- Crescimento rápido de ~60% para ~99%
- Estabilização próxima aos 100% nas últimas épocas
- Indicativo de aprendizado eficiente das características visuais

### Matriz de Confusão

<img width="866" height="773" alt="image" src="https://github.com/user-attachments/assets/a863dd4c-04c8-4848-a74f-d525304849a1" />


**Classes Identificadas:**
1. **Diamond_Print_Halter_Cami_Dress**: 13/14 corretas (92.8%)
2. **Chic_Sweater_Crop_Top**: 7/9 corretas (77.8%)
3. **Cutout_A-Line_Dress**: 7/8 corretas (87.5%)
4. **Contrast-Paneled_Henley**: 7/8 corretas (87.5%)
5. **Dip-Dyed_Knotted_Sweater**: 8/11 corretas (72.7%)

**Observações:**
- **Melhor Performance**: Diamond_Print_Halter_Cami_Dress (apenas 1 erro)
- **Maior Confusão**: Entre suéteres (Chic_Sweater vs Dip-Dyed_Knotted)
- **Precisão Geral**: Excelente capacidade de distinção entre categorias
- **Erro Principal**: Confusão entre tipos similares de suéteres

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

## 10. Integração com LLMs para Recomendação Interativa

### Arquitetura Proposta

Este sistema de classificação pode ser integrado com LLMs para criar um **assistente de moda inteligente** e interativo:

```
[Imagem do Usuário] → [ViT Classifier] → [Embeddings + Metadata] → [LLM] → [Recomendações Personalizadas]
```

### Componentes da Integração

#### 1. **Pipeline de Classificação**
```python
# Exemplo de integração
def classify_and_extract_features(image):
    # Classificação com ViT
    category = vit_model.predict(image)
    
    # Extração de embeddings visuais
    visual_features = vit_model.get_features(image)
    
    # Análise de atributos
    attributes = extract_attributes(image)  # cor, padrão, estilo
    
    return {
        'category': category,
        'visual_embedding': visual_features,
        'attributes': attributes
    }
```

#### 2. **Prompt Engineering para LLM**
```python
def create_fashion_prompt(classification_result, user_context):
    prompt = f"""
    Você é um consultor de moda especializado. 
    
    ITEM ANALISADO:
    - Categoria: {classification_result['category']}
    - Atributos: {classification_result['attributes']}
    
    CONTEXTO DO USUÁRIO:
    - Ocasião: {user_context['occasion']}
    - Estilo preferido: {user_context['style']}
    - Orçamento: {user_context['budget']}
    
    Forneça recomendações personalizadas de:
    1. Peças complementares
    2. Combinações de cores
    3. Acessórios adequados
    4. Dicas de styling
    """
    return prompt
```

#### 3. **Base de Conhecimento Enriquecida**
```json
{
  "Diamond_Print_Halter_Cami_Dress": {
    "style_tags": ["feminino", "casual-chic", "verão"],
    "occasions": ["casual", "encontros", "passeios"],
    "color_palette": ["neutros", "pastéis"],
    "complementary_items": ["jaqueta jeans", "sandálias", "bolsa pequena"],
    "styling_tips": "Ideal para clima quente, combine com acessórios delicados"
  }
}
```

### Casos de Uso Práticos

#### **1. Personal Stylist Virtual**
```
Usuário: *[envia foto de um vestido]*
Sistema: "Identifico um Diamond Print Halter Cami Dress! Para um look casual-chic, 
           recomendo combinar com uma jaqueta jeans clara e sandálias nude. 
           Que tipo de ocasião você tem em mente?"
```

#### **2. Montagem de Looks Completos**
```
Usuário: "Tenho uma reunião importante, como usar essa peça?"
Sistema: "Para um ambiente profissional, sugiro uma blazer estruturado por cima, 
           sapatos fechados e acessórios discretos. O estampado fica mais sério 
           com cores neutras."
```

#### **3. Shopping Inteligente**
```
Usuário: "Gostei deste estilo, me ajude a encontrar peças similares"
Sistema: "Baseado no seu Cutout A-Line Dress, você pode gostar de:
           - Vestidos evasê com detalhes únicos
           - Peças com recortes estratégicos
           - Cores sólidas em tons terrosos"
```

### Vantagens da Integração

#### **Personalização**
- Histórico de preferências do usuário
- Análise de estilo pessoal baseada em uploads
- Recomendações contextuais (clima, ocasião, humor)

#### **Aprendizado Contínuo**
- Feedback do usuário melhora as recomendações
- Atualização de tendências em tempo real
- Refinamento baseado em interações

#### **Conhecimento Especializado**
- Combinação de análise visual + conhecimento fashion
- Dicas de styling profissionais
- Educação sobre moda e tendências

### Implementação Técnica

#### **Backend Architecture**
```python
class FashionAssistant:
    def __init__(self):
        self.vit_classifier = load_vit_model()
        self.llm_client = OpenAI()  # ou outro LLM
        self.knowledge_base = load_fashion_db()
    
    def analyze_and_recommend(self, image, user_context):
        # 1. Classificação visual
        classification = self.vit_classifier.predict(image)
        
        # 2. Enriquecimento com base de conhecimento
        item_info = self.knowledge_base.get(classification['category'])
        
        # 3. Geração de prompt contextual
        prompt = self.create_contextual_prompt(classification, item_info, user_context)
        
        # 4. Consulta ao LLM
        recommendations = self.llm_client.chat(prompt)
        
        return {
            'classification': classification,
            'recommendations': recommendations,
            'styling_tips': item_info['styling_tips']
        }
```

## 11. Melhorias Futuras

### Técnicas
- [ ] Implementar data augmentation avançado
- [ ] Adicionar validação cruzada
- [ ] Suporte para mais arquiteturas (ResNet, EfficientNet)
- [ ] Otimização de hiperparâmetros automática
- [ ] Métricas de interpretabilidade do modelo

### Integração com LLMs
- [ ] Sistema de embeddings para busca semântica
- [ ] API para integração com assistentes virtuais
- [ ] Dashboard para análise de tendências
- [ ] Sistema de feedback e aprendizado ativo
- [ ] Integração com e-commerce e catálogos
- [ ] Análise de sentimento em avaliações de produtos

## Autor

- **Bruno Paiva Smit de Freitas** - [brnsmit](https://github.com/brnsmit)
