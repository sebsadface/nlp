## QQV Implementation Evaluation

### Training Steps vs. Loss
![Training Steps vs. Loss](./training_step_vs_loss.png)

### Evaluation

- **notice the long tail of PAD tokens:**  
   `[0, 7441, 64, 98, 149, 2, 9, 2697, 15, 72, 168, 171, 20, 46, 236, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]`

- **reported loss from model:** `5.74938440322876`
- **manually calculated loss:** `5.749383449554443`
- **manually calculated loss again:** `5.749383449554443`
- **perplexity:** `313.99700927734375`
---
- **loss1:** 9.392719268798828
**perplexity1:** 12000.68810007194

- **loss2:** 6.778720855712891
**perplexity2:** 878.9437089088758

- **loss3:** 7.883161544799805
**perplexity3:** 2652.244508417094

- **loss4:** 6.5366716384887695
**perplexity4:** 689.9862283566857

---
- **train perplexity:** 403.60858960572074
- **dev perplexity:** 383.07666051094384
---
- <START> A age on evidence he nothing , United States to 28 and people , which is usual turn the <UNK> Americans was revealed of spending . <STOP>

- <START> And over the House will keep Gordon tour by the department 's Hospital , being Deal pick many months were <UNK> , on its Radio <UNK> in 2011 , this extent without an <UNK> reforms , the largest frozen authority 9 at anyone at a screenplay rates , spent question . <STOP>

- <START> dramatic Ireland cents has by New candidates which could take anything by previously sent much on . <STOP>
- <START> A eyes of photos are explained the riding a build foot Wednesday out of Afghan 2008 program , 2008 . <STOP>
- <START> After Somali government of print broke out are up two documentary by shaky <UNK> - GHz from the dollar , including strikes in Wednesday : She may fly from the rural subscription . <STOP>
- <START> recall each service at hard continue a single or his Logan spokeswoman in other Republican own champion , and <UNK> , David <UNK> Open the America Clinton away driver 's years . <STOP>
- <START> recall each year , hard into a writing or his Logan spokeswoman in other Republican own champion who was reported the David fixtures . <STOP>
- <START> Jacob after repeated <UNK> lay a single ceiling that he automatic Meyer to be Stanley accidents , who has see 4.5 days . <STOP>
- <START> Mark under the other other much in Georgia 's troop job the U.S. violence calling will produce <UNK> Her driving two , run at 12 campaign , 70 . <STOP>
- <START> His day to have epidemic , tracking remaining pasta to each months and his tiny Financial Times . <STOP>