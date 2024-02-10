## Additive Attention Implementation Evaluation

### Training Steps vs. Loss
![Training Steps vs. Loss](./training_step_vs_loss.png)

### Evaluation

- **notice the long tail of PAD tokens:**  
   `[0, 7441, 64, 98, 149, 2, 9, 2697, 15, 72, 168, 171, 20, 46, 236, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]`

- **reported loss from model:** `5.6291399002075195`
- **manually calculated loss:** `5.6291399002075195`
- **manually calculated loss again:** `5.629140853881836`
- **perplexity:** `278.42254638671875`
---
- **loss1:** 9.114241600036621
**perplexity1:** 9083.742897471115

- **loss2:** 6.477777004241943
**perplexity2:** 650.5232273559228

- **loss3:** 7.895134449005127
**perplexity3:** 2684.190439007811

- **loss4:** 6.32673978805542
**perplexity4:** 559.3300834462823

---
- **train perplexity:**  405.25525184085245
- **dev perplexity:** 385.9976889984025
---
- <START> " I were only that may should not the violence only much time for something president are at particles it 's proposal . <STOP>

- <START> All of taking European band perceived <UNK> <UNK> , Antarctic and the third relatives as fans UCLA place , a profits were singer dangerous . <STOP>

- <START> Anbar , received an threat of them on by out up by much parents , reaching out the church agent . <STOP>
- <START> tunnels and earnings Bush soldiers are 20 points to reforms -- but today . <STOP>
- <START> He is include estimates of your ; stuff managed to fix to answer scandal . <STOP>
- <START> Amazon Bay 's <UNK> has Lightning High changed . <STOP>
- <START>The leaders along said all soon called his kept deer of her billion in state friends since edge against intention of diverse -- <UNK> , 25 percent . <STOP>
- <START> The musical outlook raise Nani at the medal should give the average of by reports that are certain effect in any afternoon apart for $ £ Airport in <UNK> complain at how <UNK> ... <STOP>
- <START> <UNK> questioned enough for own First Michelle does the Reed 's nomination rivers through the lively add his Saul rights . <STOP>
- <START> These has also shortage of 1991 , the police parties . <STOP>