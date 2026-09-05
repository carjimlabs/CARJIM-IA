# CARJIM-IA

IA de detecção e classificação de células em imagens de esfregaço sanguíneo. Uso didático:
o professor coloca uma imagem em `Imagens a serem analisadas/` (ou escolhe pelo app gráfico)
e recebe a mesma imagem com molduras e o nome de cada célula, entre as 7 classes exibidas:
Hemácia, Neutrófilo, Linfócito, Monócito, Eosinófilo, Basófilo e Plaqueta.

## Arquitetura: duas etapas

O sistema usa dois modelos YOLOv8 em sequência, não um único detector de 7 classes:

1. **Detector** (`models/carjim_best.pt`) — YOLOv8 de detecção, 3 classes: Hemácia (RBC),
   Leucócito (WBC) e Plaqueta (Platelets). Treinado com fotos de campo largo (várias células
   por imagem), generaliza bem para as fotos reais que o professor tira.
2. **Classificador de subtipo** (`models/wbc_classifier.pt`) — YOLOv8-cls. Cada leucócito que
   o detector encontra é recortado da imagem e passado por este segundo modelo, que devolve o
   subtipo: Neutrófilo, Linfócito, Monócito, Eosinófilo ou Basófilo.

**Por que duas etapas em vez de um único detector de 7 classes:** essa abordagem mais simples
foi tentada várias vezes e sempre colapsava. O único dataset público com leucócitos já
rotulados por subtipo (Raabin-WBC) só tem recortes de uma célula centralizada e ampliada — um
detector treinado com isso aprende a esperar leucócitos grandes e centralizados, e falha em
fotos de campo largo com várias células pequenas. Toda tentativa de misturar as duas escalas
(pseudo-rótulos, colagem de recortes em fotos de campo largo) trocava um problema por outro.
Separar "achar a célula" (etapa 1, dataset de campo largo) de "dizer que subtipo é" (etapa 2,
recortes ampliados) resolve isso e mantém a detecção de hemácia/plaqueta idêntica ao modelo já
validado. Se o classificador de subtipo não estiver presente, os leucócitos ainda aparecem
detectados, só que como "Leucócito" genérico em vez do subtipo.

## Como usar (nesta máquina ou em qualquer outra com GPU NVIDIA)

Basta copiar a pasta `CARJIM-IA` inteira para a máquina de destino. O código detecta
automaticamente se há GPU (`cuda`) disponível; caso não haja, cai para CPU.

```
pip install -r requirements.txt
```

### Etapa 1 — detector (RBC/WBC/Platelets)

```
python scripts/01_download_dataset.py        # BCCD -> data/bccd/raw/
python scripts/02_prepare_dataset.py          # Pascal VOC -> YOLO, split treino/val, dataset.yaml
python scripts/06_merge_txlpbc.py             # opcional, mas recomendado (ver abaixo)
python scripts/07_add_platelet_crops.py       # opcional, ver abaixo
python scripts/08_merge_roboflow_allidb.py    # opcional, ver abaixo
python scripts/03_train.py                    # treina, gera models/carjim_best.pt
```

1. **01_download_dataset.py** — baixa e extrai o dataset público BCCD para `data/bccd/raw/`.
2. **02_prepare_dataset.py** — converte as anotações Pascal VOC XML para o formato YOLO,
   faz o split treino/validação e gera `data/bccd/dataset.yaml`.
3. **03_train.py** — treina o YOLOv8 (fine-tuning, continuando de `models/carjim_best.pt`
   se ele já existir, ou a partir de `yolov8s.pt` na primeira vez) usando a GPU quando
   disponível, com `imgsz=1280` (preserva células pequenas em fotos de campo largo). Ao
   final, copia o melhor checkpoint para `models/carjim_best.pt`.
4. **05_add_wide_field_sample.py** — uso opcional/contínuo. Se uma foto real (ex.: celular
   no microscópio) não for bem detectada por ter um zoom diferente do dataset, este script
   gera pseudo-rótulos para ela automaticamente (dividindo a imagem em recortes pequenos,
   rodando o modelo atual em cada um, e juntando os resultados) e a adiciona ao treino:
   `python scripts/05_add_wide_field_sample.py "caminho/da/foto.jpg"`. Depois rode
   `03_train.py` de novo. Sempre confira o preview gerado em
   `data/bccd/pseudo_label_previews/` antes de confiar nos rótulos.
5. **06_merge_txlpbc.py** — baixa o [TXL-PBC](https://github.com/lugan113/TXL-PBC_Dataset)
   (1.260 imagens, ~18k caixas, integra BCCD + 3 outras fontes públicas) e mescla no
   treino, remapeando as classes automaticamente. Deixa a detecção de leucócito/plaqueta
   bem mais robusta. Rode antes do `03_train.py`.
6. **07_add_platelet_crops.py** — integra `platelet.zip` (2.348 recortes com uma plaqueta
   centralizada cada, sem anotação) ao treino. Como não há caixas prontas, o script gera
   as caixas automaticamente por segmentação de cor (a plaqueta é roxo/violeta escuro e
   saturado, bem diferente da hemácia rosa clara ao redor) — captura tanto a plaqueta
   central quanto fragmentos/aglomerados extras que apareçam no mesmo recorte. Salva uma
   prévia com as caixas desenhadas para cada imagem em
   `data/bccd/platelet_crop_previews/` (mais um `_contact_sheet.jpg` com uma amostra) —
   confira antes de treinar; se a segmentação tiver saído ruim, apague os arquivos com
   prefixo `pltcrop_` de `images/train` e `labels/train`.
7. **08_merge_roboflow_allidb.py** — integra um export do Roboflow ALL_IDB (caixas reais de
   WBC/Plaqueta) ao treino. Esse export não tem hemácias anotadas, então só as caixas de
   WBC/Plaqueta são mescladas (ver o docstring do script para o motivo de não gerar hemácias
   por pseudo-rótulo aqui).

### Etapa 2 — classificador de subtipo de leucócito

```
python scripts/12_build_wbc_classifier_data.py  # zips Raabin -> data/wbc_cls/<split>/<Classe>/*.jpg
python scripts/13_train_wbc_classifier.py       # treina, gera models/wbc_classifier.pt
```

- **12_build_wbc_classifier_data.py** — organiza os zips do
  [Raabin-WBC](https://raabindata.com/free-data/) num `ImageFolder` (uma pasta por subtipo,
  já dividido em treino/validação).
- **13_train_wbc_classifier.py** — treina um YOLOv8-cls (`imgsz=128`) nesse `ImageFolder`,
  balanceando as classes raras por duplicação, e salva `models/wbc_classifier.pt`.

### Rodando a análise

```
python scripts/04_watch_and_infer.py
```

Fica monitorando a pasta `Imagens a serem analisadas/`; a cada imagem nova, roda a detecção,
classifica cada leucócito encontrado pelo subtipo, desenha as molduras com o nome da célula em
português e salva o resultado em `Imagens analisadas/`. Pressione `Ctrl+C` para parar. Além da
passada principal na imagem inteira, roda uma passada extra em recortes pequenos só para
reforçar a detecção de **plaquetas** (muito menores que hemácia/leucócito, por isso somem numa
única passada em fotos de campo largo) — o tamanho esperado da plaqueta é calculado como uma
fração do tamanho das hemácias detectadas na mesma imagem, então isso se ajusta
automaticamente ao zoom da foto.

## Uso interativo (app gráfico)

Além do `04_watch_and_infer.py` (monitora uma pasta), há um app gráfico pra analisar
imagens avulsas, escolhendo na hora quais classes mostrar:

```
python app.py
```

Marque as classes desejadas, clique em **Selecionar Imagem...** para escolher um arquivo do
computador, e o resultado aparece na tela. Desmarcar/marcar uma classe depois de processar a
imagem só filtra o que já foi detectado (instantâneo, não roda o modelo de novo) — exceto
marcar **Plaqueta** numa imagem que foi processada com ela desmarcada, que dispara só o
reforço de plaqueta sob demanda. **Salvar como...** exporta a imagem anotada em resolução
plena (por padrão em `Imagens analisadas/`).

Ainda não existe um `.exe` empacotado — por enquanto é preciso rodar via
`python app.py` com as dependências de `requirements.txt` instaladas.

## Classes exibidas

| Etapa | Classe interna | Rótulo exibido |
|---|---|---|
| Detector | RBC | Hemácia |
| Detector + Classificador | WBC → Neutrophil | Neutrófilo |
| Detector + Classificador | WBC → Lymphocyte | Linfócito |
| Detector + Classificador | WBC → Monocyte | Monócito |
| Detector + Classificador | WBC → Eosinophil | Eosinófilo |
| Detector + Classificador | WBC → Basophil | Basófilo |
| Detector | Platelets | Plaqueta |

Sem o classificador de subtipo presente, todo leucócito aparece como "Leucócito" genérico em
vez de um dos 5 subtipos acima.

## Dataset

- [Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset) (licença MIT), ~364
  imagens de esfregaço sanguíneo anotadas em Pascal VOC. Base do detector (etapa 1).
- [TXL-PBC](https://github.com/lugan113/TXL-PBC_Dataset), opcional via `06_merge_txlpbc.py`
  — 1.260 imagens adicionais já em formato YOLO, integrando BCCD + BCDD + PBC + Raabin-WBC.
  Também usado no detector (etapa 1).
- [Raabin-WBC](https://raabindata.com/free-data/), via `12_build_wbc_classifier_data.py` —
  única fonte pública com leucócitos rotulados por subtipo; fornece recortes de uma célula
  centralizada e ampliada, usados para treinar o classificador (etapa 2).

`data/` não é versionado no git (é grande e 100% regenerável) — rode os scripts `01` a `08` e
depois `12`/`13` para reconstruí-lo. `07_add_platelet_crops.py` e `08_merge_roboflow_allidb.py`
dependem de zips de origem (`platelet.zip`, export do Roboflow ALL_IDB) que também não estão
no repositório e precisam ser obtidos separadamente antes de rodá-los.

## Limitação conhecida: escala/zoom da foto

Essa limitação afeta apenas o **detector** (etapa 1) — o classificador de subtipo (etapa 2)
trabalha sobre um recorte já isolado do leucócito, então é bem menos sensível ao zoom da foto
original.

O detector foi treinado majoritariamente com imagens "de perto" (poucas células grandes por
foto, como o BCCD e o TXL-PBC). Fotos tiradas com bem mais zoom out (celular no microscópio,
muitas células pequenas na mesma foto) tendem a funcionar bem para **hemácias** (padrão
simples, generaliza fácil). **Leucócitos** e **plaquetas** já melhoraram bastante com o
dataset TXL-PBC e o reforço de detecção por recortes (rodado em `04_watch_and_infer.py`), mas
ainda podem sair com confiança mais baixa ou perder alguns casos em fotos com zoom muito
diferente do dataset de treino. Se isso acontecer, use `05_add_wide_field_sample.py` com fotos
reais nesse mesmo zoom para ensinar o detector a essa escala também — é a forma mais confiável
de melhorar ainda mais.
