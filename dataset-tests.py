from src.data.fed_cc_news import get_fed_cc_news
from src.data.agnews import get_agnews_data
from src.data.three_multi import get_three_multi_data
from src.data.github_wiki import get_github_wikitext_data
from src.data.wikitext import get_wikitext_data
#from src.data.wikitext_split import get_split_multi_data

print('cc news...')

get_fed_cc_news()
print('agnews mixed...')
get_agnews_data("mixed")
print('agnews specific...')
get_agnews_data("specific")
print('3multi specific...')
get_three_multi_data("specific")
print('3multi mixed')
get_three_multi_data("mixed")
print('wikitext specific...')
get_github_wikitext_data("specific")
print('wikitext mixed')
get_github_wikitext_data("mixed")
print('wikitext...')
get_wikitext_data()
'''print('split multi fr...')
get_split_multi_data("fr")
print('split multi it...')
get_split_multi_data("it")
print('split multi de...')
get_split_multi_data("de")
print('split multi en...')
get_split_multi_data("en")'''
    
