# Eigensolver 1D

Este es un eigensolver para la ecuación de Schrödinger independiente del tiempo en una dimensión, escrito con diferencias finitas. La idea es sencilla: discretizo el espacio en una malla, construyo el Hamiltoniano como una matriz (energía cinética vía un stencil de 5 puntos para la segunda derivada, más energía potencial como matriz diagonal), y diagonalizo con `scipy.sparse.linalg.eigsh` para sacar los eigenvalores (energías) y eigenvectores (funciones de onda) más bajos. Nada de esto es nuevo en el sentido de que es el método de diferencias finitas de toda la vida para resolver Schrödinger, pero lo construí y derivé yo desde cero, incluyendo el stencil de la segunda derivada, para entender bien qué está pasando en cada paso antes de usar la caja negra de scipy.

Lo escribí para explorar potenciales que no tienen solución analítica limpia, y para tener algo con lo que pudiera comparar resultados numéricos contra los que sí conozco de memoria: pozo infinito y oscilador armónico. El código soporta 9 potenciales (oscilador armónico, anarmónico, pozo infinito, pozo finito, potencial lineal, Coulomb suavizado en 1D, pozo cuártico simple, delta de Dirac discreta, y pozo doble cuártico), todos definidos en `Eigensolver_1Dimension.py` junto con la función `Schrodinger_solver` que hace el trabajo pesado. La derivación completa del stencil de diferencias finitas y del espectro analítico de cada potencial está en `Eigensolver_1Dimensional.pdf`, son mis notas a mano de cuando lo armé.

## Oscilador armónico

Es el caso de prueba obligado: potencial cuadrático, espectro exactamente equiespaciado, $E_n = \hbar\omega(n+\tfrac12)$. Aquí están las primeras seis eigenfunciones, cada una desplazada verticalmente a su propia energía y superpuestas sobre $V(x)$:

![Eigenfunciones del oscilador armónico](figures/wavefunctions_harmonic.png)

Y el diagrama de niveles correspondiente:

![Niveles de energía del oscilador armónico](figures/energy_levels_harmonic.png)

## Pozo doble cuártico

Este es el caso que de verdad me gusta enseñar, porque no tiene solución cerrada y muestra algo físicamente interesante: $V(x) = V_0(x^2-a^2)^2$ tiene dos mínimos, y cada nivel del pozo individual se desdobla en un par casi degenerado por efecto túnel a través de la barrera central. Entre más abajo esté el par respecto a la barrera, más pequeño el desdoblamiento.

![Eigenfunciones del pozo doble](figures/wavefunctions_doublewell.png)

![Niveles de energía del pozo doble](figures/energy_levels_doublewell.png)

Con $a=1.5$ y $V_0=7$, el par fundamental ($n=0,1$) se desdobla por apenas $5.8\times10^{-6}$, mientras que el segundo par ($n=2,3$), más arriba y más cerca de la barrera, se desdobla por $8.9\times10^{-4}$. Justo lo que uno esperaría de efecto túnel: entre más energía tiene el estado respecto a la barrera, más fácil cruza, más grande el desdoblamiento.

## Validación contra soluciones analíticas

Para el pozo infinito ($L=10$, $\hbar=m=1$, $E_n = n^2\pi^2\hbar^2/2mL^2$) y el oscilador armónico ($\omega=m=\hbar=1$, $E_n=\hbar\omega(n+\tfrac12)$) comparé los eigenvalores numéricos contra los analíticos:

```
Pozo infinito (N=2000)                    Oscilador armónico (N=2000)
 n     numeric     analytic   rel.err      n     numeric     analytic   rel.err
 1   0.04935566  0.04934802  1.548e-04     0   0.50000000  0.50000000  8.623e-11
 2   0.19742264  0.19739209  1.548e-04     1   1.50000000  1.50000000  1.995e-10
 3   0.44420095  0.44413220  1.548e-04     2   2.50000000  2.50000000  4.267e-10
 4   0.78969057  0.78956835  1.548e-04     3   3.50000000  3.50000000  7.696e-10
 5   1.23389152  1.23370055  1.548e-04     4   4.49999999  4.50000000  1.225e-09
 6   1.77680378  1.77652879  1.548e-04     5   5.49999999  5.50000000  1.796e-09
```

El oscilador armónico da un error relativo de 9 a 10 órdenes de magnitud, básicamente precisión de máquina. El pozo infinito se queda estancado en ~$1.5\times10^{-4}$ para todos los niveles, un error plano que no crece con $n$. Esa planitud tiene una explicación concreta, no es ruido: viene de una decisión de diseño consciente que tomé al construir el stencil, y vale la pena explicarla porque es la parte más interesante del proyecto en términos numéricos.

## El límite de precisión en la frontera, y por qué lo dejé así

El stencil de 5 puntos para la segunda derivada es de cuarto orden en el interior de la malla. Pero en las dos filas adyacentes a cada frontera de Dirichlet, el stencil completo necesita un punto que queda un paso más allá del borde del dominio, un punto que no existe. Deriví a mano las fórmulas correctas de un solo lado (one-sided) para esas filas, están en la página 2 de `Eigensolver_1Dimensional.pdf`. El problema es que si las uso ahí y dejo el stencil central en el resto, la matriz del Hamiltoniano deja de ser simétrica: el coeficiente que una fila one-sided le asigna a su vecino no coincide con el coeficiente que esa vecina, con stencil central, le asigna de regreso. Rompí $H=H^\dagger$, y con eso pierdo la garantía de eigenvalores reales y eigenvectores ortogonales que es literalmente el punto de resolver un problema de Hermitian eigenvalue.

Así que dejé el stencil central uniforme en toda la malla, tal como lo justifiqué en mis notas originales. El costo es que esas dos filas por frontera pasan de $O(dx^4)$ a $O(dx^2)$ de precisión local, y eso limita la convergencia global a $O(dx)$, no $O(dx^4)$, para cualquier estado con amplitud o pendiente apreciable en la frontera. Lo confirmé barriendo $N$ para el pozo infinito:

![Convergencia del pozo infinito](figures/convergence_isw.png)

El error medido cae exactamente como $1/N$, no como $1/N^4$. Esto es consistente en cada punto con la referencia $O(1/N)$. El oscilador armónico no muestra este problema porque su función de onda ya decayó a prácticamente cero mucho antes de llegar a la frontera del dominio, así que el defecto en esas dos filas casi no tiene con qué interactuar. Un arreglo correcto que preserve la simetría exacta existe (operadores de frontera tipo summation-by-parts con una cuadratura no uniforme), pero es harina de otro costal y lo dejé fuera de este proyecto a propósito.

## Qué hay en el repo

`Eigensolver_1Dimension.py` es el módulo con el solver, los 9 potenciales, y las funciones de graficado (`plot_wavefunctions`, `plot_energy_levels`). `validate_1d.py` corre la comparación contra soluciones analíticas y regenera la gráfica de convergencia. `demo_figures.py` regenera las cuatro figuras del oscilador armónico y el pozo doble. `Eigensolver_1Dimension.ipynb` es el notebook de demo con todo corrido y las gráficas ya incrustadas. `Eigensolver_1Dimensional.pdf` son mis notas de la derivación completa del stencil y del espectro analítico de los 9 potenciales.

Para correrlo:

```
pip install -r requirements.txt
python demo_figures.py
python validate_1d.py
```

## Alcance

Esto es 1D nada más, con diferencias finitas de orden bajo-medio y diagonalización dispersa estándar (sin shift-invert, sin aceleración para dominios grandes). No es un paquete pensado para producción ni para mallas enormes, es la herramienta que armé para entender espectros de potenciales 1D y para practicar el puente entre la derivación en papel y el código. La versión 2D de este mismo enfoque vive en `Eigensolver_2Dimensions.py`.
