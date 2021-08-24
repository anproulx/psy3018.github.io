---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
(irmf-chapitre)=
# IRM fonctionnelle

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/anproulx">
        <img src="https://avatars.githubusercontent.com/u/65092948?v=4?s=100" width="100px;" alt=""/>
        <br /><sub><b>Andréanne Proulx</b></sub>
      </a>
      <br />
        <a title="Contenu">🤔</a>
    </td>
    <td align="center">
      <a href="https://github.com/pbellec">
        <img src="https://avatars.githubusercontent.com/u/1670887?v=4?s=100" width="100px;" alt=""/>
        <br /><sub><b>Pierre bellec</b></sub>
      </a>
      <br />
        <a title="Contenu">🤔</a>
    </td>
  </tr>
</table>

```{warning}
Ce chapitre est en cours de développement. Il se peut que l'information soit incomplète, ou sujette à changement.
```

## Objectifs du cours

- Comprendre les bases **physiques** et **physiologiques** du signal BOLD
- Comprendre l'hypothèse du **système linéaire**, invariant dans le temps
- Connaître les principales étapes de **pré-traitement** des données IRMf
- Connaître le principe de génération d'une **carte d'activation**

Dans ce chapitre, vous serez initié aux fondements théoriques qui nous permettent de faire la construction de cartes d'activation en IRM fonctionnelle, menant à des inférences sur l'organisation fonctionnelle du cerveau. Nous débuterons ce chapitre en contrastant tout d'abord les objectifs respectifs de l'IRM structurelle vus précédemment, et de l'IRM fonctionnelle. 

#### En IRM structurelle
`L'objectif` poursuivi par l'IRM structurelle est la mise en valeur des **structures et propriétés des tissus** du cerveau. L'acquisition d'un volume (3D) s'étalent typiquement sur plusieurs minutes.
        
#### En IRM fonctionnelle (T2*)
`L'objectif` poursuivi est la mise à découvert de **l'organisation fonctionnelle** du cerveau. Nous sommes intéressés au dynamisme de l'activité neuronale, mesurée indirectement via l'activité **BOLD**, et ce, souvent en relation avec des manipulations expérimentales. Comparativement à la modalité d'IRM structurelle, c'est une **séquence** de volumes qui est acquise (4D) sur la durée de l'acquisition.

-----------------------------------------------------------------
#### 📋 Résumé 📋

|               |   `IRM structurelle`     | `IRM fonctionnelle (T2*)`  |
| ------------- |:-------------:| -----:|
| `Objet d'étude`      | **Anatomie** et propriétés des tissus | Organisation **fonctionnelle**|
| `Dimension`     | **3D**   |   **4D** |
| `Durée de l'acquisition` | Plusieurs minutes |  Durée d'acquisition plus courte |

------------------------------------------------------------------
```{code-cell} ipython 3
:tags: ["hide-input"]
# @ cells to hide
# import 
import pandas as pd
import nilearn
import numpy as np
from nilearn.plotting import plot_stat_map, plot_anat, plot_img
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from nilearn import datasets
from PIL import Image, ImageDraw
from nilearn.input_data import NiftiMasker
import pylab as plt
import PIL
from PIL import Image
glue("t1-fig", fig, display=False)
```
```{note}
Here is a note!
```

def main():
    return

## Couplage neurovasculaire
```{code-cell} ipython 3
:tags: ["hide-input"]

from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/SWOww_Ensqg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

## Modèle de réponse hémodynamique
```{code-cell} ipython 3
:tags: ["hide-input"]

from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/91JVOAwOOyE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

### Prétraitement des données d'IRMf
```{code-cell} ipython 3
:tags: ["hide-input"]

from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/-u0WCHGi8_A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

### Cartes d'activation
```{code-cell} ipython 3
:tags: ["hide-input"]

from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/ayr6xFw2qPs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

### Conclusions et références suggérées

### Références

```{bibliography}
:filter: docname in docnames
```

## Exercices

### Exercice 1
Choisissez la bonne réponse. Des données d’IRMf sont en général...
 1. Une image du cerveau.
 2. Une dizaine d’images du cerveau.
 3. Des centaines d’images du cerveau.

### Exercice 2
Qu’est ce que le signal BOLD? (vrai / faux).
 1. Un signal très courageux.
 2. Une séquence pondérée en T2*.
 3. Un type de séquence d’IRM qui mesure l’activité du cerveau.
 4. Un type de séquence d’IRM qui mesure l’oxygénation du sang.

### Exercice 3
Choisissez la bonne réponse. Le signal BOLD dépend sur...
 1. Le flux sanguin local.
 2. Le volume sanguin local.
 3. La concentration relative en désoxyhémoglobine.
 4. Toutes les réponses ci-dessus.

### Exercice 4
Vrai / faux. Le principe d’additivité de la réponse hémodynamique est...
 1. Un modèle mathématique.
 2. Une propriété bien connue du couplage neurovasculaire.
 3. Une hypothèse courante, en partie confirmée expérimentalement.

### Exercice 4
Quel est l’événement principal à l’origine des changements de signal mesuré par le BOLD?

### Exercice 5
Dans quelle portion de l’arbre vasculaire observe-t-on les changements principaux liés à l’activité neuronale locale?

### Exercice 6
Vrai / faux?
 1. La réponse hémodynamique démarre immédiatement après l’excitation neuronale.
 2. La réponse hémodynamique est visible une seconde après l’excitation neuronale.
 3. La réponse hémodynamique est maximale 2 secondes après l’excitation neuronale.
 4. La réponse hémodynamique est toujours visible 7 secondes après l’excitation neuronale.
 5. La réponse hémodynamique est toujours visible 30 secondes après l’excitation neuronale.

### Exercice 7
Vrai / faux / peut-être? (expliquez pourquoi)
 1. Les données IRM fonctionnelle et structurelle doivent être réalignées pour générer une carte d’activation.
 2. Les données d’IRMf “brutes” (avant prétraitement) sont inutilisables pour générer une carte d’activation.
 3. Le lissage spatial est important, même pour une analyse individuelle.

### Exercice 8
On compare l’activation pour une tâche de mémoire dans le cortex frontal entre deux groupes de participants: des sujets sains et des sujets âgés (N=20 par groupe). Contrairement à nos hypothèses, on ne trouve aucune différence. Donnez trois raisons qui peuvent expliquer ce résultat.
