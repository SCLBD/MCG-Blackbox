# MCG: Generalizable Black-box Adversarial Attack with Meta Learning.

<!---
## [Overview](#overview)

<a href="#top">[Back to top]</a>
-->

<img src="docs/pipeline.pdf" width="800px"/> 

#### Abstract

In the scenario of black-box adversarial attack, the target model's parameters are unknown, 
and the attacker aims to find a successful adversarial perturbation based on query feedback under a query budget. 
Due to the limited feedback information, existing query-based black-box attack methods often require many queries for 
attacking each benign example. To reduce query cost, we propose to utilize the feedback information across historical 
attacks, dubbed example-level adversarial transferability. Specifically, by treating the attack on each benign example 
as one task, we develop a meta-learning framework by training a meta generator to produce perturbations conditioned on 
benign examples. When attacking a new benign example, the meta generator can be quickly fine-tuned based on the feedback 
information of the new task as well as a few historical attacks to produce effective perturbations. Moreover, since the 
meta-train procedure consumes many queries to learn a generalizable generator, we utilize model-level adversarial 
transferability to train the meta generator on a white-box surrogate model, then transfer it to help the attack against 
the target model. The proposed framework with the two types of adversarial transferability can be naturally combined 
with any off-the-shelf query-based attack methods to boost their performance.

#### Requirements



#### Attack

#### Train

#### Citation

If interested, you can read our recent works about backdoor learning, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).

```
@article{wu2022backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao and Zha, Hongyuan},
  journal={arXiv preprint arXiv:2206.12654},
  year={2022}
}

@inproceedings{dbd-backdoor-defense-iclr2022,
title={Backdoor Defense via Decoupling the Training Process},
author={Huang, Kunzhe and Li, Yiming and Wu, Baoyuan and Qin, Zhan and Ren, Kui},
booktitle={International Conference on Learning Representations},
year={2022}
}

@inproceedings{ssba-backdoor-attack-iccv2021,
title={Invisible backdoor attack with sample-specific triggers},
author={Li, Yuezun and Li, Yiming and Wu, Baoyuan and Li, Longkang and He, Ran and Lyu, Siwei},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={16463--16472},
year={2021}
}
```


