Here are **all 9 formulas** with detailed variable explanations in terms of **beauty trend analysis**:

---

### **1. Hawkes Process — Measuring Self-Sustaining Momentum in Comment Activity**

$$
\lambda_i(t) = \mu_i + \sum_{t_j < t} \alpha_{ij} \, \kappa_\phi(t - t_j)
$$

* $\lambda_i(t)$: **Comment generation rate** for creator $i$ at time $t$ - measures viral momentum
* $\mu_i$: **Baseline comment activity** - natural engagement without influence
* $\alpha_{ij}$: **Influence strength** from creator $j$ to $i$ - cross-content impact
* $\kappa_\phi(t - t_j)$: **Temporal decay function** - recent comments have stronger viral effects

$$
\mathcal{R}_{c,t} = \rho \big( \mathbf{A}_c(t) \big)
$$

* $\mathcal{R}_{c,t}$: **Reproduction number** - measures if a beauty trend is self-sustaining (>1 = viral)
* $\rho(\cdot)$: **Spectral radius** - maximum eigenvalue indicating trend propagation strength
* $\mathbf{A}_c(t)$: **Excitation matrix** - cross-influence between different beauty creators/content

---

### **2. Topological Burst Index (TBI) — Detecting Novelty in Conversations Before Volume Spikes**

$$
\mathrm{TBI}_{c,t} = 
\frac{\sum_k \big( d_k - b_k \big)_{t} \;-\; \sum_k \big( d_k - b_k \big)_{t-1}}
{\sum_k \big( d_k - b_k \big)_{t-1} + \epsilon}
$$

* $\mathrm{TBI}_{c,t}$: **Structural novelty score** - detects new conversation patterns before volume spikes
* $b_k, d_k$: **Birth/death times** of discussion topics - when new beauty concepts emerge/disappear
* $(d_k - b_k)$: **Persistence lifetime** - how long beauty topics remain relevant in discussions
* $\epsilon$: **Smoothing constant** - prevents division by zero in stability calculation

---

### **3. Fusion Score — Combining Fundamentals and Momentum Into a Robust Trend Indicator**

$$
\tilde{\alpha}_{c,t} = \frac{r_{c,t} - \mathbb{E}[r_{c,t} \mid \mathcal{F}_{t-1}]}{\sqrt{\mathrm{Var}(r_{c,t} \mid \mathcal{F}_{t-1})}}
$$

* $\tilde{\alpha}_{c,t}$: **Risk-adjusted trend surprise** - measures unexpected beauty trend performance
* $r_{c,t}$: **Raw trend metric** (e.g., engagement rate) for category $c$ at time $t$
* $\mathbb{E}[\cdot]$: **Expected value** - baseline trend expectation from historical data
* $\mathcal{F}_{t-1}$: **Information set** - all available data up to previous time period

$$
\mathrm{FusionScore}_{c,t} = w_1 \cdot \tilde{\alpha}_{c,t} + w_2 \cdot \log(1 + \mathcal{R}_{c,t}) + w_3 \cdot \mathrm{TBI}_{c,t}
$$

* $\mathrm{FusionScore}_{c,t}$: **Comprehensive trend strength** - unified beauty trend indicator
* $w_1, w_2, w_3$: **Weight coefficients** - balance between surprise, momentum, and novelty

---

### **4. TF-IDF-Based Trend Scoring — Quantifying Term Importance in Beauty Content**

$$
\mathrm{tfidf}_{t,d} = \mathrm{tf}_{t,d} \cdot \log \left( \frac{N}{\mathrm{df}_t} \right)
$$

* $\mathrm{tfidf}_{t,d}$: **Term importance score** - identifies key beauty keywords in content
* $\mathrm{tf}_{t,d}$: **Term frequency** - how often beauty term $t$ appears in document $d$
* $\mathrm{df}_t$: **Document frequency** - number of beauty videos/discussions mentioning term $t$
* $N$: **Total documents** - complete beauty content corpus size

$$
\mathrm{trend\_score}(t) = \sum_{d=1}^{N} \mathrm{tfidf}_{t,d}
$$

* $\mathrm{trend\_score}(t)$: **Aggregate trend importance** - overall significance of beauty concept $t$

---

### **5. Market Gap Detection — Identifying Under-Served Product Segments**

$$
\mathrm{GapScore}_p = \mathrm{freq}(p) \cdot \left(1 - \mathrm{LoR}(p)\right) \cdot \mathrm{novelty}(p)
$$

* $\mathrm{GapScore}_p$: **Market opportunity score** - identifies profitable beauty product gaps
* $\mathrm{freq}(p)$: **Mention frequency** - how often product concept $p$ is discussed
* $\mathrm{LoR}(p)$: **L'Oréal presence indicator** - flags if brand already serves this segment
* $\mathrm{novelty}(p)$: **Innovation potential** - uniqueness of the product concept

$$
\mathrm{LoR}(p) = \begin{cases}
1 & \text{if } p \text{ contains L'Oréal brand terms} \\
0 & \text{otherwise}
\end{cases}
$$

---

### **6. Ingredient Trend Momentum — Measuring Rise in Ingredient Popularity**

$$
\mathrm{Momentum}_i(t) = \frac{\mathrm{count}_i(t) - \mathrm{count}_i(t-\Delta t)}{\mathrm{count}_i(t-\Delta t) + \epsilon}
$$

* $\mathrm{Momentum}_i(t)$: **Growth rate** - measures rising popularity of beauty ingredient $i$
* $\mathrm{count}_i(t)$: **Ingredient mentions** in current time period for ingredient $i$
* $\Delta t$: **Time comparison window** - typically 30-90 days for beauty trends
* $\epsilon$: **Numerical stability** - prevents division errors

$$
\mathrm{TrendScore}_i = \mathrm{Momentum}_i(t) \cdot \mathrm{Frequency}_i
$$

* $\mathrm{TrendScore}_i$: **Comprehensive ingredient trend** - combines growth and popularity

---

### **7. Content Similarity Clustering — Grouping Related Beauty Concepts**

$$
\mathrm{similarity}(d_i, d_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \cdot \|\mathbf{v}_j\|}
$$

* $\mathrm{similarity}(d_i, d_j)$: **Content relatedness** - measures similarity between beauty videos/content
* $\mathbf{v}_i, \mathbf{v}_j$: **TF-IDF vectors** - numerical representations of beauty content
* $\cdot$: **Dot product** - captures shared beauty terminology/concepts

$$
\mathrm{Cohesion}_C = \frac{2}{|C|(|C|-1)} \sum_{i,j \in C, i<j} \mathrm{similarity}(d_i, d_j)
$$

* $\mathrm{Cohesion}_C$: **Cluster tightness** - how related beauty concepts are within a trend group
* $|C|$: **Cluster size** - number of beauty videos/concepts in trend category

---

### **8. Semantic Innovation Index — Measuring Novelty in Product Concepts**

$$
\mathrm{InnovationIndex}(p) = \frac{|\mathrm{unique\_terms}(p)|}{|\mathrm{total\_terms}(p)|} \cdot \left(1 - \frac{\mathrm{common\_terms}(p)}{|\mathrm{terms}(p)|}\right)
$$

* $\mathrm{InnovationIndex}(p)$: **Novelty score** - measures uniqueness of beauty product concept $p$
* $\mathrm{unique\_terms}(p)$: **Distinctive keywords** - rare/innovative beauty terminology used
* $\mathrm{common\_terms}(p)$: **Standard vocabulary** - overused beauty marketing terms
* $\mathrm{terms}(p)$: **Total concept vocabulary** - all words describing the beauty product

---

### **9. Consumer Interest Heatmap — Spatial-Temporal Trend Analysis**

$$
\mathrm{Interest}_{c,t} = \alpha \cdot \log(1 + \mathrm{mentions}_{c,t}) + \beta \cdot \mathrm{sentiment}_{c,t} + \gamma \cdot \mathrm{engagement}_{c,t}
$$

* $\mathrm{Interest}_{c,t}$: **Consumer attention level** - overall interest in beauty category $c$ at time $t$
* $\mathrm{mentions}_{c,t}$: **Discussion volume** - raw comment/review count for beauty trend
* $\mathrm{sentiment}_{c,t}$: **Emotional response** - positive/negative feelings toward beauty concept
* $\mathrm{engagement}_{c,t}$: **Interaction intensity** - likes/shares per beauty content piece
* $\alpha, \beta, \gamma$: **Weight parameters** - balance between volume, sentiment, and engagement

---