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

-----------------------------------------------------------------

|               |   `IRM structurelle`     | `IRM fonctionnelle (T2*)`  |
| ------------- |:-------------:| -----:|
| `Objet d'étude`      | **Anatomie, structures et propriétés des tissus** | **Organisation fonctionnelle**|
| `Dimension`     | 1 volume - **3D**   |  Plusieurs volumes dans le temps - **4D** |
| `Durée de l'acquisition` | Plusieurs minutes |  Durée d'acquisition plus courte |

------------------------------------------------------------------

#### En IRM fonctionnelle (T2*)

**Un premier aperçu**

`L'objectif` de l'IRM fonctionnelle est l'investiguation de **l'organisation fonctionnelle** du cerveau. Lorsque nous employons cette modalité, nous sommes intéressés par le dynamisme de l'activité neuronale, et ce, souvent en relation avec des manipulations expérimentales. Comparativement à la modalité d'IRM structurelle qui visait à imager les structures du cerveau de manière statique, nous faisons ici l'acquisition d'une **séquence** de volumes (4D), et s'ajoute la dimension du temps. 

Il est possible de faire un parallèle entre les images obtenues avec l'IRM fonctionnelle et un film cinématographique. En effet, un film est composé d'une séquence d'images se succédant à une fréquence donnée donnant l'illusion de dynamisme. Avec l'IRM fonctionnelle, toutefois, plutôt que des images, c'est un volume 3D que l'on obtient si l'on s'arrête à un moment de la séquence.

```{figure} ./irm_fonctionnelle/anatomical_image_grille.png
---
width: 800px
name: image-anatomique-fig
---
Pour chaque voxel du cerveau (volume 3D), ici grossièrement délimités par la grille superposée à l'image, nous détenons plusieurs points de mesure de l'activité cérébrale dans le temps, ce qui forme ce que l'on appelle une **série temporelle** ou **décours temporel**. Une série temporelle contient l'information relative aux changements de cette activité cérébrale. Notez que cette figure a été générée à l'aide de la librairie nilearn et du Python Imaging Library (PIL). 
```


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

```{À retenir}
L'IRM fonctionnelle est une modalité d'imagerie 4D. Une différence fondamentale avec l'IRM structurelle, est l'ajout de la dimension temporelle. Ceci entraîne ultimement un compromis avec la résolution spatiale qui tend à diminuer pour un volume, pour une plus courte durée d'acquisition.
```

```
\begin{align}
{Volume(3D)}+{Temps}
\end{align}
```
## Bases physiques et physiologiques

### Couplage neurovasculaire
```{code-cell} ipython 3
:tags: ["hide-input"]

from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/SWOww_Ensqg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```
L'IRM fonctionnelle et les inférences faites sur l'organisation fonctionnelle du cerveau reposent sur une série d'hypothèses ou présuppositions théoriques. L'une de ces hypothèse importante est le phénomène du **couplage neurovasculaire**. 

```{figure} ./irm_fonctionnelle/Hypothèses_github.png
---
width: 800px
name: hypothèses-fig
---
Sur cette figure, nous pouvons voir la série d'hypothèses sur laquelle repose la modalité d'IRM fonctionnelle et la génération de cartes d'activation. Le couplage neurovasculaire concerne la partie représentant les neurones et la réponde hémodynamique. Ceux-ci seront abordés plus en détails dans la section qui suit.  
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
