This repository contains supporting materials for my talk "[Still parsing User Agent Strings for your models? Use this instead!](https://github.com/WhyR2020/abstracts/blob/master/text_mining/still_parsing_user_agent_strings_for_your_models_use_this_instead.md)" presented at two events in 2020:

* [Why R? Conference](https://2020.whyr.pl/) (see file `mastitsky_why_r_conference.html` in the project's root directory);
* [EARL Conference](https://info.mango-solutions.com/earl-online-2020) (this deck is slightly different, see file `mastitsky_earl_conference.html`).

I have also published a supplementary [article at Medium](https://mastitsky.medium.com/still-parsing-user-agent-strings-for-your-machine-learning-models-use-this-instead-8928c0e7e74f) on this topic.

Here is the abstract of that talk:

"User-Agent strings (UAS) are header fields in HTTP requests used to identify the device and browser making the request. By parsing a UAS, one can extract many data points (type and make of the visitor’s device, operating system and its version, etc.) that can be used as valuable inputs for Machine Learning models (e.g., encoding customer affluence and tech savviness). However, UAS are lacking standardised formatting, which makes their parsing a formidable task, requiring highquality regular expressions. In addition, the variety of values one can encounter in UAS is astronomically large and constantly growing, making one-hot encoding of the respective features impractical. In this talk, I will demonstrate how these problems can be overcome by embedding UAS into a low-dimensional space using Natural Language Processing techniques. I will also provide an overview of the respective R packages and illustrate the use of UAS embeddings in unsupervised and supervised learning applications."

Two example R scripts stored in the `scripts` folder can be used to create unsupervised and supervied [fastText](https://fasttext.cc/docs/en/support.html)-based transformers for UAS. To run these scripts, one can use a sample dataset provided in the folder `data` (200K unique UAS extracted from the [whatismybrowser.com](https://www.whatismybrowser.com/) database).

<hr>

I provide Data Science consulting services. [Get in touch!](mailto:sergey@nextgamesolutions.com)