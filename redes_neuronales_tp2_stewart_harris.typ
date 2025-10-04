
#import "@preview/basic-report:0.3.1": *
#import "@preview/dashy-todo:0.1.2": todo

#set math.equation(numbering: none)

#show: it => basic-report(
  doc-category: [Redes neuronales #emoji.brain],
  doc-title: "Trabajo práctico 2: \nDinámica de sistemas acoplados",
  author: "María Luz Stewart Harris",
  affiliation: "Instituto Balseiro",
  language: "es",
  compact-mode: true,
  heading-font: "Vollkorn",
  heading-color: black,
  it
)

= Dos neuronas de modelo Hodgkin y Huxley acopladas

Se simularon dos neuronas de modelo Hodgkin y Huxley acopladas mediante el siguiente sistema de 10 ecuaciones diferenciales:
#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
$
  dot(V_1) &= (I - I_("syn",1) - I_("ion",1)) / C \
  dot(m_1) &= (m_(∞)(V_1) - m_1) / (τ_(m)(V_1)) \
  dot(h_1) &= (h_(∞)(V_1) - h_1) / (τ_(h)(V_1)) \
  dot(n_1) &= (n_(∞)(V_1) - n_1) / (τ_(n)(V_1)) \
  dot(s_1) &= (s_∞(V_2) - s_1) / τ 
$,
$
  dot(V_2) &= (I - I_("syn",2) - I_("ion",2)) / C \
  dot(m_2) &= (m_(∞)(V_2) - m_2) / (τ_(m)(V_2)) \
  dot(h_2) &= (h_(∞)(V_2) - h_2) / (τ_(h)(V_2)) \
  dot(n_2) &= (n_(∞)(V_2) - n_2) / (τ_(n)(V_2)) \
  dot(s_2) &= (s_∞(V_1) - s_2) / τ 
$
)

donde 
$ s_∞(V) &=  0.5 V(1+tanh(V)) \
  tau &= 3"ms" $
== Desfasaje entre neuronas

En caso de dos neuronas excitatorias, el desfasaje entre los disparos se estabiliza en $0$ (ver #ref(<fig:HH_exc_desfasaje>)) mientras que en el caso de las neuronas inhibitorias el desfasaje se estabiliza en $pi$ (ver #ref(<fig:HH_inh_desfasaje>)).

#figure(
  image("figs/1A1 gsyn=0.5_time-1.png", width: 75%),
  caption: [Simulación de dos neuronas de modelo Hodgkin y Huxley acopladas con interacción excitatoria ($g_"syn" = 0.5$).]
)<fig:HH_exc_desfasaje>

#figure(
  image("figs/1C1 gsyn=0.5_time-1.png", width: 75%),
  caption: [Simulación de dos neuronas de modelo Hodgkin y Huxley acopladas con interacción inhibitoria ($g_"syn" = 0.5$).]
)<fig:HH_inh_desfasaje>

La figura #ref(<fig:HH_exc_desfasaje_vs_tiempo>) muestra la evolución del desfasaje entre dos neuronas en el sistema excitatorio para distintos valores de $g_"syn"$. Para todos los casos, el desfasaje tiende a $0$. Mientras más alto es $g_"syn"$, más rápido se acerca al desfasaje $0$.

#figure(
  image("figs/Vsyn=0mV - I=10uA_offset_time.png", width: 75%),
  caption: [Simulación del desfasaje entre dos neuronas de modelo Hodgkin y Huxley acopladas con interacción excitatoria ($g_"syn" = 0.5$).]
)<fig:HH_exc_desfasaje_vs_tiempo>


== Tasa de disparo del sistema acoplado como función de $g_("syn")$

Para el sistema excitatorio, la frecuencia del sistema una vez que alcanza el estado estacionario decrece al aumentar el parámetro $g_"syn"$, tal como se observa en la #ref(<fig:f_vs_gsyn_exc>). Para $g_"syn"$ entre $0$ y $1$ toma valores entre $83$Hz y $77$Hz aproximadamente. En el caso inhibitorio, la frecuencia se mantiene aproximadamente constante en $82.5plus.minus 1"Hz"$

#figure(
  image("figs/Vsyn=0mV - I=10uA_f_vs_gsyn.png", width: 75%),
  caption: [Frecuencia de disparo para sistema de dos neuronas excitatorias en función de $g_"syn"$.]
)<fig:f_vs_gsyn_exc>

#figure(
  image("figs/Vsyn=-80mV - I=10uA_f_vs_gsyn.png", width: 75%),
  caption: [Frecuencia de disparo para sistema de dos neuronas inhibitorias en función de $g_"syn"$.]
)<fig:f_vs_gsyn_inh>

= Dos poblaciones de neuronas acopladas

Se considera el siguiente sistema de poblaciones de neuronas excitatorias e inhibitorias:

$ tau (dif h_e)/(dif t) = -h_e + g_("ee")f_e - g_("ei")f_i + I_e \ 
tau (dif h_i)/(dif t) = -h_i + g_("ie")f_e - g_("ii")f_i + I_e 
$
donde 

$
  f_e &= h_e H(h_e) \
  f_i &= h_i H(h_i) \
  H(x) &= cases(1 "si" x>0, 0 "si" x <=0)\
  g_"ee", g_"ei", g_"ie", g_"ii" &> 0
$

El sistema tiene similitudes con el modelo de Cowan-Wilson con promediado temporal (_time coarse graining_).#footnote[Wilson, H.R.; Cowan, J.D. (1972). "Excitatory and inhibitory interactions in localized populations of model neurons". Biophys. J. 12 ] Al igual que en ese modelo, se interpretan las variables $h_i$ y $h_e$ como la proporción de neuronas que disparan por unidad de tiempo de la población inhibitoria y excitatoria respectivamente. Es importante notas que $h_i=0$ o $h_e=0$ no implica que no hay actividad en la población de neuronas inhibitorias o excitatorias, sino que se interpreta como la actividad de base de bajo nivel de la población. En la misma línea, un pequeño valor negativo en $h_i$ o $h_e$ representan una baja en la actividad de la población con respecto a su actividad de base. 


== Puntos fijos y estabilidad

Para todos los casos, 
 - Se buscan la $h_x$ nulclinas igualando $(dif h_x)/(dif t)=0$ para $x=i,e$
 - Se obtienen los puntos fijos $Y^*=vec(h_e, h_i)$ calculando la intersección entre las nulclinas
 - Se calcula para que condiciones el punto fijo obtenido está dentro del caso analizado
- Se definen las funciones $f_e$ y $f_i$:

$ f_(e)(h_e, h_i) = (dif h_e)/(dif t)(h_e, h_i) $
$ f_(i)(h_e, h_i) = (dif h_i)/(dif t)(h_e, h_i) $

- Se define la matrix $AA$:

$ AA=lr(mat((dif f_e)/(dif h_e), (dif f_e)/(dif h_i); (dif f_i)/(dif h_e), (dif f_i)/(dif h_i))|)_((h_e, h_i) = Y^*) $

- Se calcula $"Tr"(AA)$ y $lr(|AA|)$ y se concluye que el punto fijo es estable si $"Tr"(AA) < 0$ y $lr(|AA|) > 0$ 


=== Caso 1: $h_e, h_i <=0$

==== Punto fijo
$ cases(
  0 = -h_e + I_e  & #"   " (h_e "nulclina"),
  0 = -h_i + I_i  & #"   " (h_i "nulclina"),
) $

$=> Y^* = vec(-I_e, -I_i)$ es un punto fijo siempre que se cumpla que:


$ cases(h_e &= -I_e <=0 ,
  h_i &= -I_i <=0) \
  => cases( I_e > 0,
  I_i > 0) $

==== Estabilidad del punto fijo

$ f_(e)(h_e, h_i) = -h_e + I_e $
$ f_(i)(h_e, h_i) = -h_i + I_i $

$ AA = mat(-1, 0; 0, -1) $

$ "Tr"(AA) = -2 < 0 $
$ lr(|AA|) = 1 > 0 $

$=>$ El punto fijo es estable 

=== Caso 2: $h_e<=0, h_i >0$

==== Punto fijo

$ cases(
  0 = -h_e -g_"ei" h_i+ I_e,
  0 = -(g_"ii"+1)h_i + I_i
) $
$ cases(
  h_e = -g_"ei" h_i+ I_e  & #"   " (h_e "nulclina"),,
  h_i = I_i/(g_"ii"+1)  & #"   " (h_i "nulclina"),
) $
$=> Y^* = vec(-g_"ei"/(g_"ii"+1)I_i + I_e, 1/(g_"ii"+1)I_i)$ es un punto fijo siempre que se cumpla que:


$ cases(h_e &= -g_"ei"/(g_"ii"+1)I_i + I_e &<=0,
  h_i &= 1/(g_"ii"+1)I_i &> 0 ) \
  => cases( I_e &<=  g_"ei"/(g_"ii"+1)I_i,
  1/(g_"ii"+1)I_i &> 0) $

==== Estabilidad del punto fijo

$ f_(e)(h_e, h_i) = -h_e -g_"ei" h_i+ I_e $
$ f_(i)(h_e, h_i) = -(g_"ii"+1)h_i + I_i $

$ AA = mat(-1, -g_"ei"; 0, -(g_"ii"+1)) $

$ "Tr"(AA) = -g_"ii"-2 < 0 $
$ lr(|AA|) = g_"ii"+g_"ei"+1 > 0 $

$=>$ El punto fijo es estable 

=== Caso 3: $h_e>0, h_i <=0$

==== Punto fijo

$ cases(
  0 = (g_"ee"-1)h_e + I_e,
  0 = g_"ie"h_e - h_i + I_i
) $

$ cases(
  h_e = I_e/(1-g_"ee") & #"   " (h_e "nulclina"),
  h_i = g_"ie"h_e+I_i & #"   " (h_i "nulclina")
) $

$=> Y^* = vec(1/(1-g_"ee")I_e , g_"ie"/(1-g_"ee")I_e + I_i)$ es un punto fijo siempre que se cumpla que:

$
  cases(
    h_e = I_e/(1-g_"ee") > 0,
    h_i = g_"ie"/(1-g_"ee")I_e+I_i <= 0
  )\
  => cases(
    I_e/(1-g_"ee") > 0,
    I_i <= g_"ie"/(g_"ee"-1)I_e
  )
$

==== Estabilidad del punto fijo

$ 
  f_(e)(h_e, h_i) &= (g_"ee"-1)h_e + I_e \
  f_(i)(h_e, h_i) &= g_"ie"h_e - h_i + I_i
$

$ AA = mat(g_"ee"-1, 0; g_"ie", -1) $

$ "Tr"(AA) = g_"ee"-2 "   "(<0 " si " g_"ee"<2) $
$ lr(|AA|) = 1-g_"ee" "   "(>0 " si " g_"ee">1) $

$=>$ El punto fijo es estable si $1 < g_"ee" < 2$


=== Caso 4: $h_e,h_i >0$

==== Punto fijo

$ cases(
  0 = (g_"ee"-1)h_e -g_"ei" h_i+ I_e,
  0 = (g_"ie")h_e -(g_"ii"+1)h_i + I_i
) $

$ cases(
  h_e = g_"ei"/(g_"ee"-1)h_i -1/(g_"ee"-1) I_e& #"   " (h_e "nulclina"),,
  h_i = g_"ie"/(g_"ii"+1)h_e +1/(g_"ii"+1) I_i& #"   " (h_i "nulclina"),
) $

$=> Y^* = vec(
  (1+g_"ii")/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_e -  g_"ei"/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_i,
  g_"ie"/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_e + (1-g_"ee")/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_i
)= vec(
  (1+g_"ii")I_e -  g_"ei"I_i,
  g_"ie"I_e + (1-g_"ee")I_i
) dot 1/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))$ es un punto fijo siempre que:

$
  cases(
    h_e = (1+g_"ii")/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_e -  g_"ei"/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_i > 0,
    h_i = g_"ie"/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_e + (1-g_"ee")/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_i > 0
  )
$

==== Estabilidad del punto fijo
$ 
  f_(e)(h_e, h_i) = (g_"ee"-1)h_e -g_"ei" h_i+ I_e,
  f_(i)(h_e, h_i) = (g_"ie")h_e -(g_"ii"+1)h_i + I_i
$

$ AA = mat(g_"ee"-1, -g_"ei"; g_"ie", -(g_"ii"+1)) $

$ "Tr"(AA) = g_"ee"+g_"ii"-2 "   "(<0 " si " g_"ee"+g_"ii"<2) $
$ lr(|AA|) = -(g_"ee"-1)(g_"ii"+1) + g_"ei"^2 "   "(>0 " si " g_"ei"^2>(g_"ee"-1)(g_"ii"+1)) $

$=>$ El punto fijo es estable si: 
$cases(
g_"ei"^2>(g_"ee"-1)(g_"ii"+1),
g_"ee"+g_"ii"<2
)$

== Simulación

La #ref(<fig:pobl_sim_caso_1>) muestra una simulación de las poblaciones de neuronas acopladas para el caso 1 ($h_e, h_i <=0$). La figura muestra que $h_e$ y $h_i$ tienden al punto fijo correspondiente:

$ h_e = -I_e = -1\
  h_i = -I_i = -1 $ 

Notar que se cumplen las condiciones para que ese punto sea un punto fijo:

$ I_e = 1 > 0\
  I_i = 1 > 0 $ 


#figure(
  image("figs/2A1 I_e=I_i=-1 he0=-0.2 hi0=-0.1_time.png", width: 75%),
  caption: [Simulación de dos poblaciones de neuronas acopladas. \
  Caso 1.\
  ($I_e = I_i = -1, g_"ii"=g_"ie"=g_"ei"=g_"ee"=1, h_("i0") = -0.2, h_("e0") = -0.1$)]
)<fig:pobl_sim_caso_1>

La #ref(<fig:pobl_sim_caso_2>) muestra un ejemplo del caso 2, ya que $h_"e0" <= 0, h_"i0" > 0$. El punto fijo correspondiente al caso 2 es:

$
  h_e &= -g_"ei"/(g_"ii"+1)I_i + I_e &=& (-2)/(1+1)dot 2 + 1 &= -1 <= 0 \
  h_i &= 1/(g_"ii"+1)I_i &=& 1/(1+1) dot 2 &= 1 > 0
$

#figure(
  image("figs/2A2 I_e=1 I_i=2 he0=2 hi0=-5_time.png", width: 75%),
  caption: [Simulación de dos poblaciones de neuronas acopladas. \
  Caso 2.\
  ($I_e = 1, I_i = 2, g_"ii"=g_"ie"=g_"ei"=1, g_"ee"=2, h_("i0") = 2, h_("e0") = -5$)]
)<fig:pobl_sim_caso_2>

La #ref(<fig:pobl_sim_caso_4>) muestra un ejemplo del caso 4, ya que $h_"e0",h_"i0" > 0$. El punto fijo correspondiente al caso 4 es:

$
  h_e &= (1+g_"ii")/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_e - g_"ei"/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_i &>& 0 \
  h_i &= g_"ie"/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_e + (1-g_"ee")/(g_"ie"g_"ei"+(1-g_"ee")(1+g_"ii"))I_i &>& 0
$

Como $g_"ii"=g_"ie"=g_"ei"=g_"ee"=I_e=I_i=1$:

$
  h_e &= 2/1 dot 1 - 1/1 dot 1 &= 1&>& 0 \
  h_i &= 1/1 dot 1 - 0 dot 1   &= 1&>& 0
$

#figure(
  image("figs/2A4 I_e=1 I_i=1 he0=1 hi0=5_time.png", width: 75%),
  caption: [Simulación de dos poblaciones de neuronas acopladas. \
  Caso 4.\
  ($g_"ii"=g_"ie"=g_"ei"=g_"ee"=I_e=I_i=1,h_"e0"=5,h_"i0"=1$)]
)<fig:pobl_sim_caso_4>

= Anexo
El repostorio de código utilizado para simular y graficar se encuentra en: #box[#link( "https://github.com/malustewart/redes-neuronales-tp2" )]
