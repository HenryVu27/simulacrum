# Umwelt: related work and novelty brief

Date: 2026-11-04. Every bibliography entry was confirmed against a primary record (arXiv abstract page, publisher or proceedings page, Crossref DOI record, or official repository). Anything not confirmed is marked UNVERIFIED.

## Bottom line

No prior work found runs the core design: one fixed predictive learner, many sensory and bodily configurations, identical world trajectories, one concept battery, and a crossed manipulation of world physics that splits concept variance into body and world shares. Every ingredient has precedent. The novelty is the controlled factorial design and the decomposition, not any single method. The most dangerous neighbor is a 2026 position paper with the same name and thesis: Bosch, Sommers, Doerig and Kietzmann, "The Umwelt Representation Hypothesis". It reports no experiments, so this project can be framed as its first controlled test, set against the Platonic Representation Hypothesis (PRH) of Huh et al. 2024.

## 1. Bibliography

The verified bibliography is at the end of this file: requested items, novelty neighbors, then datasets and tools.

## 2. Novelty check

Searched: embodied artificial intelligence (AI), sensory substitution in reinforcement learning (RL), multimodal world models, developmental robotics and sensorimotor-contingency work, artificial life, controlled rearing, emergent communication between heterogeneous agents, representational convergence, and variance partitioning, through early November 2026.

### Closest prior work

| # | Work | What it did | What umwelt adds |
|---|------|-------------|------------------|
| 1 | Das et al. 2020 | Probed predictive agents in one 3D world with a frozen question-answering decoder (objects, colors, counts, relations). Compared learning objectives. | Closest method template, inverted: fix the learner, vary the body, add violation-of-expectation (VoE) and cross-agent tests. |
| 2 | Wijmans et al. 2023 | "Blind" agents sensing only egomotion. Occupancy maps decoded from memory. | One deprived configuration, task reward, one world. Umwelt sweeps senses under a reward-free predictive objective and varies physics. |
| 3 | Ostrovski, Castro and Dabney 2021 | "Tandem RL", modeled on Held and Hein: a passive twin makes identical updates on the active twin's data, and fails. | Tandem scores control. Umwelt's yoked observer is a world model without action signals, scored on self-versus-other causal attribution. |
| 4 | Laflaquière and Garcia Ortiz 2019; Philipona, O'Regan and Nadal 2003 | Spatial structure recovered from sensorimotor prediction whatever the sensor wiring. | Closest philosophical precedent (structure from invariants), but single toy agents and space only. |
| 5 | Pandey, Lee, Wood and Wood 2024 | "Digital twin" controlled rearing: networks raised on the same visual worlds as newborn chicks. | Varies experience, not senses. Umwelt varies senses under one experience and adds a physics axis. |
| 6 | Tiwary et al. 2025 | Co-evolved eyes and brains; task decides eye type. | Varies sensors under selection, scores behavior and morphology. No concept probes or physics manipulation. |
| 7 | Piloto et al. 2022; Garrido et al. 2025 | VoE surprise shows learned models acquire permanence, solidity and continuity (PLATO; video joint-embedding predictive architecture, V-JEPA). | Single visual channel, passive video. Umwelt reuses the protocol across bodies and across lawful and lawless worlds. |
| 8 | Huh et al. 2024; Bosch et al. 2026; Koepke et al. 2026 | PRH predicts cross-modal convergence; the Umwelt hypothesis predicts alignment tracks shared ecological constraints; Koepke et al. find only coarse convergence. | Pretrained models or position papers. Umwelt is a controlled experiment that can separate the claims. |
| 9 | Wang 2026, Social-JEPA (withdrawn) | Independent world models from different viewpoints had near-isometric latents. | Viewpoint only, and withdrawn. Useful precedent for translation maps, not evidence. |
| 10 | Conwell et al. 2024 | Partitioned variance across architecture, objective and data for hundreds of vision models. | Closest learner-versus-data split. No body or physics axis, no agents. |

Also adjacent: Tang and Ha 2021 (sensor reshuffling, not sensor identity); Lampinen et al. 2023 (passive data can teach active causal strategies); Jayaraman and Grauman 2015 (ego-motion as a learning signal, framed by the kitten carousel); Gregor et al. 2019 (decoding position from belief states); Evans et al. 2021 (a Kantian symbolic learner); Pitzer and Mihai 2026 (agents with different modalities cannot communicate without fine-tuning); Hagen and Gaudl 2024 (topology from sensorimotor prediction; a reported active-versus-passive comparison is UNVERIFIED).

### Honest novelty claim

As far as these searches show, umwelt is the first controlled study in which one fixed predictive learner is raised with many sensory and bodily configurations on identical trajectories, crossed with manipulated physics, scored on one concept battery (linear probes, VoE surprise, translation maps, centered kernel alignment or CKA), with variance split into body, world and interaction shares. It is not already done. It is a recombination of established parts, so reviewers will judge design rigor, not method novelty.

### Design risks

1. With shared trajectories no agent chooses its actions. The yoked contrast is access to an action signal (an efference copy), not agency. Say so, or add a self-driven condition.
2. The body-by-world interaction may dominate, leaving "shares" ill-defined. Report a two-way analysis of variance (ANOVA) with the interaction, or a Shapley decomposition.
3. Decodability is not use. Add causal tests modeled on concept injection (Lindsey 2025).
4. Alignment scores inflate with scale and pairing (Koepke et al. 2026). Use the oracle and memoryless agents as bounds, plus permutation baselines.
5. Cite Bosch et al. 2026 in the first paragraph because of the name collision.

## 3. Open data and weights for the frontier phase

| Resource | Content | License or access |
|---|---|---|
| Open X-Embodiment | 1M+ robot trajectories, 22 embodiments, 60 datasets, Reinforcement Learning Datasets (RLDS) format | Software Apache 2.0, other materials CC BY 4.0. TensorFlow Datasets (TFDS) or `gs://gdm-robotics-open-x-embodiment`. Cite each dataset. |
| DROID | About 76,000 Franka trajectories, 350 hours, 564 scenes, stereo cameras | TFDS at `gs://gresearch/robotics`. License UNVERIFIED. |
| Ego4D | 3,000+ hours egocentric video | Signed data use agreement. Commercial terms UNVERIFIED. |
| Ego-Exo4D | 1,286 hours; audio, gaze, point clouds, camera poses, inertial measurement unit (IMU) | Separate EgoExo license agreement. |
| Project Aria (Nymeria, Aria Everyday Activities) | Egocentric multisensor glasses data | Nymeria reported CC BY-NC 4.0 (search results, page not opened). Aria Everyday Activities terms UNVERIFIED. |
| EgoPet | Internet-sourced egocentric animal video, mostly dogs and cats | License UNVERIFIED. |
| DECADE (Ehsani et al. 2018) | 380 dog-mounted camera clips plus IMUs | Code public. Data license UNVERIFIED. |
| Bio-logger Ethogram Benchmark (BEBE) | 9 annotated animal-borne tag datasets | Public on GitHub. License UNVERIFIED. |
| Movebank | Animal tracking studies | Per study: CC0, CC BY or CC BY-NC if public; owner terms if private. |
| V-JEPA 2 | Video encoders 300M to 1B parameters; action-conditioned V-JEPA 2-AC | Code mostly MIT. Hugging Face weights Apache 2.0, ungated. V-JEPA 2.1 (March 2026, up to 2B) weights license UNVERIFIED. |
| NVIDIA Cosmos-Predict2.5 | 2B and 14B video world models, robot action-conditioned variant | NVIDIA Open Model License; gated on Hugging Face. |
| NVIDIA Cosmos 3 | Cosmos3-Nano, 16B, video, audio and action, released 2026-05-31 | OpenMDW 1.1. Gating UNVERIFIED. |

## 4. Library versions and API notes (PyPI, 2026-11-04)

| Package | Version (date) | Notes |
|---|---|---|
| jax, jaxlib | 0.12.0.1 (2026-10-21) | Python 3.12+, NumPy 2.2+, AVX2 on x86-64. `pmap` now wraps `jit(shard_map)`; `PmapSharding` and `device_put_replicated` removed (0.10.0). Use `jax.set_mesh(mesh)`, not `with mesh:` (0.10.1). `jnp.empty` is uninitialized (0.11.0). `jax.random.Key` alias added; from 0.13 `random.key` returns it, so use typed keys now. |
| optax | 0.2.8 (2026-03-20) | Stable. |
| flax | 0.12.12 (2026-10-27) | NNX recommended for new code; Linen still supported. `nnx.Optimizer(model, tx, wrt=nnx.Param)`, then `optimizer.update(model, grads)` inside `nnx.jit`. |
| equinox | 0.13.8 (2026-05-05) | Alternative to NNX. |
| mujoco, mujoco-mjx | 3.17.0 (2026-10-22) | MuJoCo XLA (MJX) `impl='jax'` (default) is differentiable and runs on TPU and Apple Silicon, but mesh ray casts are slow. `impl='warp'` (`mujoco-mjx[warp]`) is most complete, has no autodiff, needs `naconmax` and `njmax` sized to the batch, and alone offers batched camera rendering. |
| playground (MuJoCo Playground) | 0.2.0 (2026-03-16) | Apache 2.0; `--impl warp`. |
| brax | 0.14.2 (2026-03-15) | Use its PPO and SAC trainers; MJX supersedes its physics. |
| craftax | 1.7.0 (2026-10-27) | MIT. Gymnax-style pure `reset(rng, params)` and `step(rng, state, action, params)`. Reference for one pytree world state with several renderers. |
| jaxmarl | 0.2.0 (2026-09-10) | Apache 2.0. Per-agent dictionaries with an `"__all__"` done key. Template for per-agent sensor functions over one state. |
| gymnax, chex, distrax, orbax-checkpoint | 1.0.1, 0.1.92, 0.1.9, 0.12.9 | Environments, testing, distributions, checkpoints. |

Build implications: write the world as a pure `step(state, action, physics_params)` so the physics regime is a `vmap` axis, and each sense as a pure function of world state so every agent reads one trajectory. Keep geometry to primitives if MJX-JAX ray casts drive vision, echoes and smell. `pyproject.toml` already pins `jax[cpu]>=0.12.0.1`, `flax>=0.12.12`, `mujoco>=3.17.0` and `optax>=0.2.8`; swap to the CUDA extra named in the JAX install docs before cloud runs.

---

## Bibliography

Venue abbreviations: ICML (International Conference on Machine Learning), NeurIPS (Conference on Neural Information Processing Systems), ICLR (International Conference on Learning Representations), CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition), ECCV (European Conference on Computer Vision), ICCV (International Conference on Computer Vision), ICRA (IEEE International Conference on Robotics and Automation), PNAS (Proceedings of the National Academy of Sciences), TPAMI (IEEE Transactions on Pattern Analysis and Machine Intelligence).

### A. Requested items

- Held, R., and Hein, A. 1963. "Movement-produced stimulation in the development of visually guided behavior." Journal of Comparative and Physiological Psychology 56(5): 872-876. https://doi.org/10.1037/h0040546. The kitten carousel: self-moved kittens developed visually guided behavior, yoked passive kittens did not. Template for the yoked observer.
- Ehsani, K., Bagherinezhad, H., Redmon, J., Mottaghi, R., and Farhadi, A. 2018. "Who Let the Dogs Out? Modeling Dog Behavior from Visual Data." CVPR 2018, pp. 4051-4060. https://arxiv.org/abs/1803.10827. Introduces the DECADE dog-mounted camera and IMU dataset. The only real-animal egocentric precedent for the dog agent.
- Huh, M., Cheung, B., Wang, T., and Isola, P. 2024. "The Platonic Representation Hypothesis." ICML 2024 (position paper, PMLR 235). https://arxiv.org/abs/2405.07987. Claims representations converge across models and modalities; notes that models with different information cannot fully converge. The hypothesis umwelt tests under controlled sensory differences.
- Garrido, Q., Ballas, N., Assran, M., Bardes, A., Najman, L., Rabbat, M., Dupoux, E., and LeCun, Y. 2025. "Intuitive physics understanding emerges from self-supervised pretraining on natural videos." arXiv:2502.11831. https://arxiv.org/abs/2502.11831. V-JEPA passes VoE tests (98% zero-shot on IntPhys) while pixel predictors and multimodal language models do not. VoE protocol and baseline.
- Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. 2023. "Mastering Diverse Domains through World Models." arXiv:2301.04104. https://arxiv.org/abs/2301.04104. DreamerV3 preprint.
- Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. 2025. "Mastering diverse control tasks through world models." Nature 640: 647-653. https://doi.org/10.1038/s41586-025-08744-2. Nature version of DreamerV3. The recurrent state-space world model is the natural candidate for the shared learning rule.
- Riochet, R., Ynocente Castro, M., Bernard, M., Lerer, A., Fergus, R., Izard, V., and Dupoux, E. 2018. "IntPhys: A Framework and Benchmark for Visual Intuitive Physics Reasoning." arXiv:1803.07616. https://arxiv.org/abs/1803.07616. Journal version: "IntPhys 2019: A Benchmark for Visual Intuitive Physics Understanding." TPAMI 44(9): 5016-5025, 2022. https://doi.org/10.1109/TPAMI.2021.3083839. Matched possible and impossible events for permanence, shape constancy and continuity.
- Bordes, F., Garrido, Q., Kao, J. T., Williams, A., Rabbat, M., and Dupoux, E. 2025. "IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments." arXiv:2506.09849. https://arxiv.org/abs/2506.09849. Permanence, immutability, continuity and solidity; most models sit near chance. Template for the impossible-event suite.
- Bear, D. M., Wang, E., Mrowca, D., Binder, F. J., Tung, H.-Y. F., et al. 2021. "Physion: Evaluating Physical Prediction from Vision in Humans and Machines." NeurIPS 2021 Datasets and Benchmarks Track. https://arxiv.org/abs/2106.08261. Shows extracting physical state from vision is the bottleneck. Supports including the oracle agent.
- Piloto, L. S., Weinstein, A., Battaglia, P., and Botvinick, M. 2022. "Intuitive physics learning in a deep-learning model inspired by developmental psychology." Nature Human Behaviour 6(9): 1257-1267. https://doi.org/10.1038/s41562-022-01394-8. PLATO (Physics Learning through Auto-encoding and Tracking Objects) and the Physical Concepts VoE dataset.
- Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. 2019. "Similarity of Neural Network Representations Revisited." ICML 2019, PMLR 97: 3519-3529. https://proceedings.mlr.press/v97/kornblith19a.html. Defines CKA, the cross-agent similarity measure.
- Schut, L., Tomasev, N., McGrath, T., Hassabis, D., Paquet, U., and Kim, B. 2025. "Bridging the human-AI knowledge gap through concept discovery and transfer in AlphaZero." PNAS 122(13): e2406675122. https://doi.org/10.1073/pnas.2406675122. Concept vectors filtered for teachability and novelty. Method for finding concepts an agent has that the probe list lacks.
- Gurnee, W., and Tegmark, M. 2024. "Language Models Represent Space and Time." ICLR 2024. https://arxiv.org/abs/2310.02207. Linear probes recover space and time coordinates. Probe design for the location and elapsed-time concepts.
- Open X-Embodiment Collaboration. 2024. "Open X-Embodiment: Robotic Learning Datasets and RT-X Models." ICRA 2024. https://arxiv.org/abs/2310.08864. Project page https://robotics-transformer-x.github.io/. Multi-embodiment robot data for the frontier phase.
- Grauman, K., et al. 2022. "Ego4D: Around the World in 3,000 Hours of Egocentric Video." CVPR 2022. https://arxiv.org/abs/2110.07058. Human egocentric multisensor data.
- Grauman, K., et al. 2024. "Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives." CVPR 2024. https://arxiv.org/abs/2311.18259. Synchronized first- and third-person views with audio, gaze and IMU data. A real-data analog of the same world seen by different observers.
- Lindsey, J. 2025. "Emergent Introspective Awareness in Large Language Models." Transformer Circuits Thread, Anthropic, 2025-10-29. https://transformer-circuits.pub/2025/introspection/index.html. Summary: https://www.anthropic.com/research/introspection. Concept injection writes a concept vector into activations. Template for causal tests of probed concepts.
- Rosas, F. E., Geiger, B. C., Luppi, A. I., Seth, A. K., Polani, D., Gastpar, M., and Mediano, P. A. M. 2024. "Software in the natural world: A computational approach to hierarchical emergence." arXiv:2402.09090 (v1 title: "...emergence in complex multi-level systems"). https://arxiv.org/abs/2402.09090. Informational, causal and computational closure of macro-level processes. A criterion for when an agent's "object" variable is a self-contained level.
- von Uexküll, J. 2010. A Foray into the Worlds of Animals and Humans, with A Theory of Meaning. Trans. J. D. O'Neil. University of Minnesota Press, Posthumanities series. ISBN 9780816659005. https://www.upress.umn.edu/9780816659005/a-foray-into-the-worlds-of-animals-and-humans/. Source of the Umwelt concept.
- Condillac, É. B. de. 1754. Traité des sensations. Open text: https://doi.org/10.1522/030166757. Verified through Falkenstein, L., and Grandi, G. "Étienne Bonnot de Condillac." Stanford Encyclopedia of Philosophy (revised 2017). https://plato.stanford.edu/entries/condillac/. The statue receives smell first, and touch later gives it external space. Model for the smell-only agent.
- Strawson, P. F. 1959. Individuals: An Essay in Descriptive Metaphysics. Chapter 2, "Sounds." Routledge edition 2025, pp. 51-78. https://doi.org/10.4324/9781003605140-4. The purely auditory "no-space world": can a hearer reidentify objective particulars? Tested by the hearing-only and smell-only agents.
- Worrall, J. 1989. "Structural Realism: The Best of Both Worlds?" Dialectica 43(1-2): 99-124. https://doi.org/10.1111/j.1746-8361.1989.tb00933.x. What survives theory change is structure. Umwelt's analog: what survives a change of senses.
- Meillassoux, Q. 2008. After Finitude: An Essay on the Necessity of Contingency. Trans. R. Brassier. Continuum (now Bloomsbury Academic). https://doi.org/10.5040/9781350252059. Critique of correlationism. The oracle agent and the world share are an operational handle on mind-independent structure.

### B. Novelty neighbors

- Das, A., Carnevale, F., Merzic, H., Rimell, L., Schneider, R., Abramson, J., Hung, A., Ahuja, A., Clark, S., Wayne, G., and Hill, F. 2020. "Probing Emergent Semantics in Predictive Agents via Question Answering." ICML 2020. https://arxiv.org/abs/2006.01016.
- Wijmans, E., Savva, M., Essa, I., Lee, S., Morcos, A. S., and Batra, D. 2023. "Emergence of Maps in the Memories of Blind Navigation Agents." ICLR 2023. https://arxiv.org/abs/2301.13261.
- Ostrovski, G., Castro, P. S., and Dabney, W. 2021. "The Difficulty of Passive Learning in Deep Reinforcement Learning." NeurIPS 2021. https://arxiv.org/abs/2110.14020.
- Laflaquière, A., and Garcia Ortiz, M. 2019. "Unsupervised Emergence of Egocentric Spatial Structure from Sensorimotor Prediction." NeurIPS 2019. https://arxiv.org/abs/1906.01401.
- Philipona, D., O'Regan, J. K., and Nadal, J.-P. 2003. "Is There Something Out There? Inferring Space from Sensorimotor Dependencies." Neural Computation 15(9): 2029-2049. https://doi.org/10.1162/089976603322297278.
- Pandey, L., Lee, D., Wood, S. M. W., and Wood, J. N. 2024. "Parallel development of object recognition in newborn chicks and deep neural networks." PLOS Computational Biology 20(12): e1012600. https://doi.org/10.1371/journal.pcbi.1012600.
- Tiwary, K., Young, A., Tasneem, Z., Klinghoffer, T., Dave, A., Poggio, T., Nilsson, D.-E., Cheung, B., and Raskar, R. 2025. "What if eye...? Computationally recreating vision evolution." Science Advances 11(51): eady2888. https://doi.org/10.1126/sciadv.ady2888.
- Bosch, V., Sommers, R., Doerig, A., and Kietzmann, T. C. 2026. "The Umwelt Representation Hypothesis: Rethinking Universality." arXiv:2604.17960. https://arxiv.org/abs/2604.17960.
- Koepke, A. S., Zverev, D., Ginosar, S., and Efros, A. A. 2026. "Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale." arXiv:2604.18572. https://arxiv.org/abs/2604.18572.
- Wang, Y. 2026. "Social-JEPA: Emergent Geometric Isomorphism." arXiv:2603.02263 (withdrawn). https://arxiv.org/abs/2603.02263.
- Conwell, C., Prince, J. S., Kay, K. N., Alvarez, G. A., and Konkle, T. 2024. "A large-scale examination of inductive biases shaping high-level visual representation in brains and machines." Nature Communications 15: 9383. https://doi.org/10.1038/s41467-024-53147-y.
- Tang, Y., and Ha, D. 2021. "The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning." NeurIPS 2021. https://arxiv.org/abs/2109.02869.
- Lampinen, A. K., et al. 2023. "Passive learning of active causal strategies in agents and language models." NeurIPS 2023. https://arxiv.org/abs/2305.16183.
- Jayaraman, D., and Grauman, K. 2015. "Learning image representations tied to ego-motion." ICCV 2015. https://arxiv.org/abs/1505.02206.
- Gregor, K., Rezende, D. J., Besse, F., et al. 2019. "Shaping Belief States with Generative Environment Models for RL." NeurIPS 2019. https://arxiv.org/abs/1906.09237.
- Evans, R., Hernández-Orallo, J., Welbl, J., Kohli, P., and Sergot, M. 2021. "Making sense of sensory input." Artificial Intelligence 293: 103438. https://doi.org/10.1016/j.artint.2020.103438.
- Pitzer, N., and Mihai, D. 2026. "Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems." arXiv:2601.22041. https://arxiv.org/abs/2601.22041.
- Hagen, O., and Gaudl, S. 2024. "The Role of Embodiment in Learning Representations: Discovering Space Topology via Sensory-Motor Prediction." Proceedings of the AAAI Symposium Series 4(1): 353-358. https://doi.org/10.1609/aaaiss.v4i1.31816. Bibliographic record and abstract verified; full text blocked, so the active-versus-passive detail is UNVERIFIED.
- Gao, R., Chen, C., Al-Halah, Z., Schissler, C., and Grauman, K. 2020. "VisualEchoes: Spatial Image Representation Learning through Echolocation." ECCV 2020. https://arxiv.org/abs/2005.01616. Echo simulation precedent for the bat agent.
- Singh, S. H., van Breugel, F., Rao, R. P. N., and Brunton, B. W. 2023. "Emergent behaviour and neural dynamics in artificial agents tracking odour plumes." Nature Machine Intelligence 5: 58-70. https://doi.org/10.1038/s42256-022-00599-w. Odor-plume simulation precedent for the dog and smell-only agents.

### C. Datasets, models and tools

- Bar, A., et al. 2024. "EgoPet: Egomotion and Interaction Data from an Animal's Perspective." ECCV 2024, LNCS pp. 377-394. https://arxiv.org/abs/2404.09991.
- Hoffman, B., Cusimano, M., et al. 2024. "A benchmark for computational analysis of animal behavior, using animal-borne tags." Movement Ecology 12. https://doi.org/10.1186/s40462-024-00511-8. Data: https://github.com/earthspecies/BEBE.
- DROID project page: https://droid-dataset.github.io/.
- Movebank data policy: https://www.movebank.org/cms/movebank-content/data-policy.
- V-JEPA 2 repository (code, checkpoints, V-JEPA 2 arXiv:2506.09985 and V-JEPA 2.1 arXiv:2603.14482 per its BibTeX): https://github.com/facebookresearch/vjepa2. Weights: https://huggingface.co/facebook/vjepa2-vitg-fpc64-256.
- NVIDIA Cosmos-Predict2.5: https://huggingface.co/nvidia/Cosmos-Predict2.5-2B. Cosmos 3: https://huggingface.co/nvidia/cosmos3-nano.
- Matthews, M., et al. 2024. "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning." ICML 2024. https://github.com/MichaelTMatthews/Craftax.
- Rutherford, A., Ellis, B., Gallici, M., et al. 2024. "JaxMARL: Multi-Agent RL Environments and Algorithms in JAX." NeurIPS 2024 Datasets and Benchmarks Track. https://arxiv.org/abs/2311.10090. Code: https://github.com/FLAIROx/JaxMARL.
- Zakka, K., et al. 2025. "MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer." https://github.com/google-deepmind/mujoco_playground.
- MJX documentation: https://mujoco.readthedocs.io/en/latest/mjx.html. JAX changelog: https://docs.jax.dev/en/latest/changelog.html. Flax documentation: https://flax.readthedocs.io/en/latest/.
