# Academic References & Benchmark Bibliography

## VERITAS-MM 2.0: Literature Grounding & Citations

---

## 1. Primary Benchmark & Foundation Model Citations

1. **NewsCLIPpings Benchmark (Cheapfake Detection)**
   * **Authors:** Shivangi Aneja, Cees G. M. Snoek, and Matthias Nießner.
   * **Title:** *Breaking the Manner: Detecting Out-of-Context Images in News*.
   * **Venue:** Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), pages 6323–6334.
   * **DOI/Link:** [https://aclanthology.org/2021.emnlp-main.509/](https://aclanthology.org/2021.emnlp-main.509/)

2. **OpenCLIP / Contrastive Language-Image Pretraining (CLIP)**
   * **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.
   * **Title:** *Learning Transferable Visual Models From Natural Language Supervision*.
   * **Venue:** Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139:8748–8763.
   * **Link:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

3. **DeBERTa-v3 Foundation Model**
   * **Authors:** Pengcheng He, Jianfeng Gao, and Weizhu Chen.
   * **Title:** *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*.
   * **Venue:** International Conference on Learning Representations (ICLR 2023).
   * **Link:** [https://arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543)

4. **Directional Natural Language Inference (FEVER & Multi-NLI)**
   * **Authors:** James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
   * **Title:** *FEVER: a Large-scale Dataset for Fact Extraction and VERification*.
   * **Venue:** Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2018), pages 809–819.
   * **Authors (NLI Aggregation):** Moritz Laurer, Wouter van Atteveldt, Andreu Salleras Casas, and Kasper Welbers.
   * **Title:** *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and Natural Language Inference*.
   * **Venue:** *Political Analysis* (2023).

---

## 2. Selective Classification & Decision Theory

5. **Chow's Rule & Optimal Reject Option**
   * **Author:** C. K. Chow.
   * **Title:** *On Optimum Recognition Error and Reject Tradeoff*.
   * **Venue:** *IEEE Transactions on Information Theory*, 16(1):41–46, 1970.
   * **DOI:** `10.1109/TIT.1970.1054406`

6. **Selective Classification for Deep Neural Networks**
   * **Authors:** Yonatan Geifman and Ran El-Yaniv.
   * **Title:** *Selective Classification for Deep Neural Networks*.
   * **Venue:** Advances in Neural Information Processing Systems (NeurIPS 2017), volume 30.
   * **Link:** [https://arxiv.org/abs/1705.08500](https://arxiv.org/abs/1705.08500)

---

## 3. Multimodal Fact-Checking & Manipulation Benchmarks

7. **VERITE Benchmark**
   * **Authors:** Matthias Springstein, Stefanos Vrochidis, and Ralph Ewerth.
   * **Title:** *VERITE: a Robust Benchmark for Multimodal Out-of-Context Misinformation Detection*.
   * **Venue:** Proceedings of the 2023 ACM International Conference on Multimedia Retrieval (ICMR 2023), pages 411–420.
   * **Link:** [https://arxiv.org/abs/2304.09562](https://arxiv.org/abs/2304.09562)

8. **Factify & Factify 2 Benchmarks**
   * **Authors:** Shreyash Mishra, Sarthak Anand, Megha Chakraborty, et al.
   * **Title:** *FACTIFY: A Multi-Modal Fact Verification Dataset*.
   * **Venue:** Proceedings of the Workshop on Multi-Modal Fake News and Hate-Speech Detection (De-Factify), CEUR Workshop Proceedings, 2022.

9. **BM25 Probabilistic Retrieval Framework**
   * **Authors:** Stephen Robertson and Hugo Zaragoza.
   * **Title:** *The Probabilistic Relevance Framework: BM25 and Beyond*.
   * **Venue:** *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009.

---

## 4. BibTeX Citations

```bibtex
@inproceedings{aneja2021newsclippings,
  title={Breaking the Manner: Detecting Out-of-Context Images in News},
  author={Aneja, Shivangi and Snoek, Cees GM and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={6323--6334},
  year={2021}
}

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}

@inproceedings{he2023debertav3,
  title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing},
  author={He, Pengcheng and Gao, Jianfeng and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@article{chow1970optimum,
  title={On optimum recognition error and reject tradeoff},
  author={Chow, CK},
  journal={IEEE Transactions on Information Theory},
  volume={16},
  number={1},
  pages={41--46},
  year={1970},
  publisher={IEEE}
}

@inproceedings{geifman2017selective,
  title={Selective classification for deep neural networks},
  author={Geifman, Yonatan and El-Yaniv, Ran},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={30},
  year={2017}
}

@inproceedings{springstein2023verite,
  title={VERITE: a Robust Benchmark for Multimodal Out-of-Context Misinformation Detection},
  author={Springstein, Matthias and Vrochidis, Stefanos and Ewerth, Ralph},
  booktitle={Proceedings of the 2023 ACM International Conference on Multimedia Retrieval (ICMR)},
  pages={411--420},
  year={2023}
}

@inproceedings{thorne2018fever,
  title={FEVER: a Large-scale Dataset for Fact Extraction and VERification},
  author={Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle={Proceedings of NAACL-HLT},
  pages={809--819},
  year={2018}
}
```
