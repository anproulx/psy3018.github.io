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

Ce quatrième chapitre introduit les fondements théoriques de l'IRM fonctionnelle menant à la construction de cartes d'activation, et ultimement à des inférences sur l'organisation fonctionnelle du cerveau. Nous aborderons dans un premier temps les **bases physiques et physiologiques** de cette modalité, qui tout comme pour l'IRM structurelle sont mises à profit afin de capturer certaines propriétés d'intérêt du cerveau. Toutefois, contraitement à l'IRM où ces propriétés se rapportaient à l'anatomie du cerveau (statique), ici, elles se rapportent plutôt au dynamisme de l'activité neuronale réflété par le **signal BOLD**. 

-----------------------------------------------------------------

|               |   `IRM structurelle`     | `IRM fonctionnelle (T2*)`  |
| ------------- |:-------------:| -----:|
| `Objet d'étude`      | **Anatomie, structures et propriétés des tissus** | **Organisation fonctionnelle**|
| `Dimension`     | 1 volume - **3D**   |  Séries de volumes dans le temps - **4D** |
| `Durée de l'acquisition` | Plusieurs minutes |  Durée d'acquisition plus courte |

------------------------------------------------------------------

> Pour un rappel concernant les acquisitions IRM, veuillez vous référer au Chapitre 2: Analyses morphométriques.

Nous verrons également **l'hypothèse du système linéaire invariant dans le temps** qui sous-tend la modélisation du signal BOLD, et permet de dériver le niveau d'activation en réponse à divers paradigmes expérimentaux. Nous introduirons ensuite les principales étapes de **prétraitement** appliquées aux images d'IRM fonctionnelle, qui, comme vous le verrez, recoupent partiellement certaines des étapes vues dans le chapitre précédent (recalage, lissage spatial). Tel que nous le verrons, ces étapes sont nécessaires afin d'éliminer le bruit pouvant s'être mêlé au signal BOLD mesuré, mais qui, en l'occurence, ne reflète pas un phénomène d'intérêt. Finalement, nous aborderons la construction de **cartes d'activation**, qui, en exploitant des concepts statistiques, permet d'émettre des hypothèses scientifiques sur l'organisation fonctionnelle du cerveau. 


### Un premier aperçu

L'IRM fonctionnelle s'est considérablement développer depuis ses débuts dans les années 1990. Aujourd'hui, il s'agit d'une méthode d'imagerie fonctionnelle largement employée par les chercheurs en raison de ses avantages vis-à-vis des autres modalités: l'IRMf est non-invasive, sécuritaire et comprend un bon compromis entre résolution temporelle et spatiale. 

`L'objectif` de l'IRMf est de cartographier les fonctions du cerveau. Pour ce faire, cette modalité acquière des images du cerveau en action (dynamique), et ce, souvent en relation avec des manipulations expérimentales, ayant été conçues pour isoler des processus cognitifs spécifiques. Elle s'appuie, comme nous le verrons, sur des séquence spécifiques, qui mène à la détection du **signal BOLD**, permettant d'inférer l'activité neuronale. 

> Les images d'IRM fonctionnelle sont un peu comme un film du cerveau. Autrement dit, elles sont composées d'une série d'images, soit ici de volumes 3D, qui se succèdent à une fréquence donnée. Si l'on s'arrête à un moment de la série, on obtient un volume 3D du cerveau. Nous pourrions acquérir des image du cerveau selon une fréquence de 1 volume à chaque 2 secondes pendant un total de 6 minutes, ce qui résulterait en une série de 180 volumes 3D du cerveau. 

>La notion de voxel 
Un volume du cerveau (3D) est formé plusieurs milliers voxels. Chaque voxel est une unité de ce volume, qui détient une coordonnée dans l'espace 3D (**x, y, z**). 

En chaque voxel du cerveau, nous détenons plusieurs points de mesure du signal BOLD dans le temps, ce qui forme ce que l'on appelle une **série temporelle** ou **décours temporel**. Typiquement, quelques dizaines à centaines de points de mesures décrivent une série temporelle (avec les appareils modernes nous parlons plutôt d'une centaine de points). Ces points de mesures sont séparées par un intervalle de temps, qui peut varier de millisecondes à quelques secondes (avec les appareils modernes on parle plutôt de l'ordre de ms, soit environ 500 ms). Comme nous le verrons plus loin, la série temporelle contient l'information relative aux changements de l'activité neuronale dans le temps. Une grande partie du travail en IRM fonctionnelle consiste à analyser de telles séries temporelles. 

```{code-cell} ipython 3
:tags: ["hide-input"]
# imports
import pandas as pd
import nilearn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from nilearn.input_data import NiftiLabelsMasker
from nilearn.plotting import plot_stat_map, plot_anat, plot_img
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from nilearn import datasets
from nilearn.input_data import NiftiMasker
glue("t1-fig", fig, display=False)
```

```{code-cell} ipython 3
:tags: ["hide-input"]

####### Generate figure below of a voxel with corresponding timeseries

# Extract time series for a subject
# load dataset
haxby_dataset = datasets.fetch_haxby()

# store functional data
haxby_func_filename = haxby_dataset.func[0]

# load brain_masker
brain_masker = NiftiMasker(
    smoothing_fwhm=6,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2,
    memory='nilearn_cache', memory_level=1, verbose=0)

# apply brain masker to extract time series
brain_time_series = brain_masker.fit_transform(haxby_func_filename,
                                               confounds=None)

# useful functions
def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

# set figure
fig = plt.figure(figsize=(10,3))

# Plot voxel
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.grid(False)
colors = np.array([[['#1f77b430']*1]*1]*1)
colors = explode(colors)
filled = explode(np.ones((1, 1, 1)))
x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

x[1::2, :, :] += 1
y[:, 1::2, :] += 1
z[:, :, 1::2] += 1

ax1.voxels(x, y, z, filled, facecolors=colors, edgecolors='white', shade=False)
plt.title("Voxel (3D")


# Plot série temporelle pour un voxel
voxel = 1
ax = fig.add_subplot(1, 2, 2)
ax.plot(brain_time_series[:, voxel])
ax.set_title("Décours temporel d'un voxel")
#plt.tight_layout()
#plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("Temps(s)", fontsize = 10)
plt.ylabel("Signal BOLD", fontsize= 10)
#fig.supxlabel('Temps (s)')
#fig.supylabel('Signal BOLD')
```

> L'IRM fonctionnelle détient une moins bonne résolution spatiale que l'IRM structurelle. En effet, nous voulons acquérir des images assez rapidement pour avoir suffisamment de mesures du signal BOLD pour arriver à estimer la réponse BOLD (qui sera abordée plus loin). Ces nouvelles considérations temporelles entraînent un compromis avec la résolution spatiale qui tend à diminuer, pour de plus courtes durées d'acquisition.

```
\begin{align}
{Volume(3D)}+{Temps}
\end{align}
```
## Bases physiques et physiologiques

### Couplage neurovasculaire
L'imagerie par résonnance magnétique fonctionnelle (IRMf) et les inférences faites sur l'organisation fonctionnelle du cerveau reposent sur une superposition d'hypothèses théoriques. L'une de ces hypothèse importante, d'origine physiologique, est le phénomène du **couplage neurovasculaire**. Ce phénomème est important comme il est exploité pour inférer l'activité neuronale, plus précisément, l'activité post-synaptique des neurones. En quoi ce phénomène consiste-t-il? Le couplage neurovasculaire décrit un **système** reliant l'**activité neuronale** à une **réponse vasculaire** caractéristique. 

> Le couplage neurovasculaire a été postulé pour la première fois vers la fin du 19e sièce dans les travaux Roy C.S. Sherrington C.S. On the regulation of the blood supply of the brain.J. Physiol. 1890; 11: 85-108. 

Mais encore? L'activité du neurone est de nature chimique : diverses molécules y opèrent, et sont impliquées dans des réactions chimiques, notamment celles menant à la production de neurotransmetteurs dans la fente synaptique. Bien que nous n'entrerons pas dans le détail de ces mécanismes, en réalité beaucoup plus complexes, ce qu'il faut retenir, c'est que lorsque le neurone est actif, cela s'accompagne de coûts. Plus précisément, il y a une augmentation de la **demande métabolique** en nutriments et en **oxygène** localement pour les neurones actifs. Cette demande métabolique entraîne une réponse vasculaire qui se caractérise par **l'augmentation du volume des capillaires** et **l'augmentation du flux sanguin**, qui **augmente l'acheminement en oxygène** vers les populations de neurones actives. Cette réponse vasculaire est fortement ciblée, puisqu'elle s'opère finement dans le cerveau (**~ 10 microns**). Elle survient à l'échelle des ramifications les plus fines de l'arbre vasculaire irriguant le cerveau, le réseau des capillaires, qui constitue sommairement la microvascularisation. Ceci confère à l'IRM fonctionelle une bonne résolution spatiale.  

La réponse vasculaire nous intéresse puisque 1) elle est couplée à l'activité neuronale et **2) elle est mesurable**. Éventuellement, nous nous retrouvons avec un surplus en oxygène à proximité des neurones actifs : l'acheminement d'oxygène (oxyhémoglobine) est plus rapide que sa consommation par les neurones (déoxyhémoglobine). L'augmentation de la concentration relative en **oxyhémoglobine** par rapport à la **déoxyhémoglobine** génère un signal, que l'on nomme le **signal BOLD** (*Blood Oxygenation Level Dependent*).

Quelle est l'origine du signal BOLD? L'hémoglobine qui fait le transport de l'oxygène existe sous deux états soit l'**oxyhémoglobine (oxygéné)** et la **déoxyhémoglobine (déoxygéné)**. Ces molécules détiennent des propriétés éléctromagnétiques ou stabilité moléculaire caractéristiques, que l'on peut exploiter lors des séquences. Autrement dit, nous tirons partie du fait que ces molécules se comportent différemment lorsque soumises au champ magnétique de l'aimant BO.

- L'**oxyhémoglobine** est **diamagnétique**
- La **déoxyhémoglobine** est **paramagnétique**

-----------------------------------------------------------------

|               |   `Déoxyhémoglobine`     | `Oxyhémoglobine`  |
| ------------- |:-------------:| -----:|
|`Propriétés électromagnétiques` |**diamagnétique**| **paramagnétique**|
| `Impact sur le signal BOLD`      | **Réduit** le signal BOLD  | **Augmente** le signal BOLD|
| `T2*`     | Décroît plus **rapidement**   |   Décroît plus **lentement** |
| `Explication` | **Ajout d'inhomogénéités/distorsions du champ** |  **Pas d'inhomogénétités du champ**  |

------------------------------------------------------------------

La déoxyhémoglobine créer des inhomogénéités du champ magnétique venant réduire le signal BOLD, alors que de son côté, l'oxyhémoglobine ne créer pas de telles inhomogénéités, et amplifie le signal BOLD. Ceci implique que lorsque les neurones s'activent, l'augmentation de la concentration relative de l'oxyhémoglobine par rapport à la dé-oxyhémoglobine localement est réflété par la **diminution des inhomogénéités** du champ, et donc, l'**augmentation du signal BOLD**. En bref, les propriétés magnétiques de l'hémoglobine selon qu'elle porte l'oxygène ou non, nous permet d'observer les changements du signal BOLD, autrement dit, les changement de la demande métabolique (oxyhémoglobine vs déoxyhémoglobine) localement, ce qui permet, éventuellement, d'inférer la réponse neuronale à une condition (présupposant un couplage neurovasculaire typique).

> Les séquences T2* * *sont sensibles aux inhomogénéités du champ. Cette information est utilisée pour reconstruire l'activité neuronale

> L'IRMf constitue par ce fait même, une **mesure indirecte** de l'activité neuronale. En effet, cette modalité ne mesure pas directement l'activité des neurones, mais plutôt la demande métabolique (sang oxygéné vs déoxygéné) associée à celle-ci.

> Le premier modèle quantitatif du couplage neurovasculaire (dit “modèle du ballon”) a été proposé par Buxton et al., MRM 1998. 

```{code-cell} ipython 3
:tags: ["hide-input"]

from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/SWOww_Ensqg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```


```{figure} ./irm_fonctionnelle/brain_capillaries.jpg
---
width: 800px
name: capillaires-fig
---
Sur cette figure, nous pouvons voir le réseau des capillaires irrigant les neurones du cerveau. Copyright © 2010 by Morgan & Claypool Life Sciences.
NCBI Bookshelf. A service of the National Library of Medicine, National Institutes of Health.
```

```{Note}
```


### La modèle de la réponse hémodynamique

**`Théorie des systèmes`**

Nous avons parlé plus tôt de l'activité neuronale et de la réponse vasculaire comme formant un **système** dans le cadre du couplage neurovasculaire. L'utilisation de ce mot n'était pas arbitraire et nous insisterons sur celui-ci dans la section qui suit. Rapportons nous dans un premier temps à la définition de ce qu'un système.

> Un système comprend un ensemble d'éléments qui interagissent selon certains principes ou règles. 

Cette définition est importante, puisqu'elle implique que nous pouvons décrire les interactions entre les éléments formant un système au moyen de fonctions ou modèles. Analoguement, la **fonction de réponse hémodynamique** ou le modèle de la réponse hémodynamique formalise la relation entre les éléments de **1) l'activité neuronale (*X*)** et **2) le signal BOLD (*y*)**, en fonction du temps. Autrement dit, elle comprend la description mathématique de ce système. 

\begin{align}
X(t) &\quad \text{intrant : activité neuronale}\\
Y(t) &\quad \text{sortie : réponse hémodynamique}\\
\end{align}

Plus tôt, nous avons parlé du couplage neurovasculaire comme figurant parmi les hypothèses théoriques (physiologiques) centrales à la modalité de l'IRM fonctionnelle. La fonction de réponse hémodynamique, et ses propriétés mathématiques, constitue une autre hypothèses importante pour les aspects de modélisation de l'IRM fonctionnelle. Essentiellement, la fonction de réponse hémodynamique sous-tend les inférences que l'on fait sur l'organisation fonctionnelle du cerveau: nous l'employons pour estimer la réponse à une tâche ou condition donnée. 

Dans la figure ci-haut, l'axe des *X* représente le temps, et l'axe de *Y*, le % du changement du signal BOLD. La ligne verticale rouge indique le début de la stimulation. La courbe bleue, pour sa part, illustre le % du changement du signal BOLD attendu suivant la stimulation. Ci-dessous sont décrit quelques caractéristiques de la fonction de réponse hémodynamique. 

```{code-cell} ipython 3
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Plot of the canonical Glover HRF
"""

import numpy as np
from nipy.modalities.fmri import hrf, utils
import matplotlib.pyplot as plt

# hrf.glover is a symbolic function; get a function of time to work on arrays

fig_BOLD = plt.plot
hrf_func = utils.lambdify_t(hrf.glover(utils.T))
t = np.linspace(0,25,200)
plt.plot(t, hrf_func(t))
a=plt.gca()
a.set_xlabel('t(s)')
a.set_ylabel('% Signal BOLD')
plt.axvline(x=0, marker = "o", color = "r")
plt.title("La fonction de réponse hémodynamique")
```

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
