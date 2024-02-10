## QQQ Implementation Evaluation

### Training Steps vs. Loss
![Training Steps vs. Loss](./training_step_vs_loss.png)

### Evaluation

- **notice the long tail of PAD tokens:**  
   `[0, 7441, 64, 98, 149, 2, 9, 2697, 15, 72, 168, 171, 20, 46, 236, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]`

- **reported loss from model:** `5.62632417678833`
- **manually calculated loss:** `5.626324653625488`
- **manually calculated loss again:** `5.626323223114014`
- **perplexity:** `277.63983154296875`
---
- **loss1:** 9.281942367553711
**perplexity1:** 10742.277115896963

- **loss2:** 6.526523590087891
**perplexity2:** 683.0196232250711

- **loss3:** 7.666393280029297
**perplexity3:** 2135.365872918697

- **loss4:** 6.333707809448242
**perplexity4:** 563.2411176940757

---
- **train perplexity:** 403.9356470955477
- **dev perplexity:** 384.30178560672005
---
- <START> The age on evidence were nothing , United bring their 28 and he was presented in usual family . <STOP>

- <START> And over the House will keep question tour by the department 's Korea , being Deal pick many months . <STOP>

- <START> dramatic \ cents in the New candidates which could important decision to previously sent much on . <STOP>
- <START> A eyes of photos are raped as not legitimacy of their negligence miles away . <STOP>
- <START> After Somali government of print broke out are up two documentary . <STOP>
- <START> Yet created Berlusconi 's even had set , handled , in our foundation . <STOP>
- <START> recall each year , hard into a writing or his Logan spokeswoman in other Republican own champion who was reported the David fixtures . <STOP>
- <START> Jacob after repeated <UNK> lay a single ceiling that he automatic Meyer -- with Stanley accidents , who has against 4.5 days . <STOP>
- <START> Mark partial the Harman is much in Georgia and troop job and growing Jim calling and more <UNK> Her driving two , run at 12 campaign , 70 <UNK> to pay a donations under al submissions for crucial signals breast instead of other orders 's ending a diet investment . <STOP>
- <START> His day of the epidemic , tracking remaining unprecedented three-day efficiency months and his tiny Financial Times . <STOP>