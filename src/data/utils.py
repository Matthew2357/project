import os
from typing import Dict, List

import numpy as np

WIKI_PATH_EN = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.en")
WIKI_PATH_FR = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.fr")
WIKI_PATH_IT = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.it")
WIKI_PATH_DE = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.de")


from .agnews import get_agnews_data
from .fed_cc_news import get_fed_cc_news
from .github_wiki import get_github_wikitext_data
from .wikitext_split import get_split_multi_data
from .three_multi import get_three_multi_data
from .wikitext import get_wikitext_data
from .wikitext_finegrained import get_wiki_multilingual
from .slim_pajama import get_slimp_dataset
from .wikimulti import get_wikimulti
from .fineweb import get_fineweb


def get_dataset(args) -> Dict[str, List[np.ndarray] | np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing up to three keys: "train" and "val", and "ref", corresponding to the tokenized training,
     validation and reference data. """

    if args.dataset == "fed_cc_news":
        return get_fed_cc_news()

    elif args.dataset == "agnews_mixed":
        return get_agnews_data("mixed")
    elif args.dataset == "agnews_specific":
        return get_agnews_data("specific")
    elif args.dataset == "three_multi_specific":
        return get_three_multi_data("specific")
    elif args.dataset == "three_multi_mixed":
        return get_three_multi_data("mixed")
    #elif args.dataset == "github_wiki_specific":
    #    return get_github_wikitext_data("specific")
    #elif args.dataset == "github_wiki_mixed":
    #    return get_github_wikitext_data("mixed")
    elif args.dataset == "wikitext":
        return get_wikitext_data()
    elif args.dataset == "wiki_split_fr":
        return get_split_multi_data("fr")
    elif args.dataset == "wiki_split_it":
        return get_split_multi_data("it")
    elif args.dataset == "wiki_split_de":
        return get_split_multi_data("de")
    elif args.dataset == "wiki_split_en":
        return get_split_multi_data("en")
    
    elif args.dataset == "slim_pajama":
        if args.dirichlet_alpha is None:
            raise NotImplementedError(f"For slim pajama, please give argument dirichlet_alpha.")
        return get_slimp_dataset(args.dirichlet_alpha, num_clients=args.num_clients, num_tokens_per_client=args.num_tokens_per_client)
        #return get_slimp_dataset(0, num_clients=args.num_clients, num_tokens_per_client=args.num_tokens_per_client)
    elif args.dataset == "wikimulti":
        if args.dirichlet_alpha is None:
            raise NotImplementedError(f"For wikimulti, please give argument dirichlet_alpha.")
        return get_wikimulti(args.dirichlet_alpha, num_clients=args.num_clients, num_tokens_per_client=args.num_tokens_per_client)
    elif args.dataset == "fineweb":
        return get_fineweb(num_clients=args.num_clients, num_tokens_per_client=args.num_tokens_per_client)

    elif "wiki_multilingual_" in args.dataset:
        return get_wiki_multilingual(args.dataset)
    
        
    else:
        raise NotImplementedError(f"Unknown dataset key {args.dataset}")


