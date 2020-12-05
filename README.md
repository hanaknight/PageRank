# Mining a Website Providing Legal Analysis of US National Security Issues

## 04 Dec 2020: Updated using Gensim word similarity
Using the ```pagerank_updated.py```, I modified the previously-developed implementation of the Google PageRank algorithm so that it also searches for the keywords in the query and the 5 most similar words. Word similarity was determined using word embeddings from the ```glove-twitter-25``` training corpus available through gensim. 
Here is the results when the search query ```drones``` was passed through:

```
INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors
INFO:root:rank=0 pagerank=0.004593398422002792 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=1 pagerank=0.0026927897706627846 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=2 pagerank=0.0020540361292660236 url=www.lawfareblog.com/roundtable-war-powers-reform   
INFO:root:rank=3 pagerank=0.0018883486045524478 url=www.lawfareblog.com/achieving-and-maintaining-cyberspace-superiority-cyber-command-and-interagency-legal-conference
INFO:root:rank=4 pagerank=0.0016097885090857744 url=www.lawfareblog.com/daniel-webster-war-powers-and-birdht
INFO:root:rank=5 pagerank=0.0012376350350677967 url=www.lawfareblog.com/civcas-reporting-responsible-command-and-feasibility
INFO:root:rank=6 pagerank=0.0011619674041867256 url=www.lawfareblog.com/slaughterbots-and-other-anticipated-autonomous-weapons-problems
INFO:root:rank=7 pagerank=0.0008982765139080584 url=www.lawfareblog.com/war-powers-red-lines-and-credibility
INFO:root:rank=8 pagerank=0.0008339908090420067 url=www.lawfareblog.com/trump-and-powers-american-presidency-part-i
INFO:root:rank=9 pagerank=0.0007972712628543377 url=www.lawfareblog.com/introducing-new-paper-weaponized-interdependence
```

## 09 Sept 2020: Using Pagerank
In this project, I attempt to replicate the link-based ranking system developed by Sergey Brin and Larry Page in their 1998 paper "The PageRank Citation Ranking: Bringing Order to the Web”. This algorithm remains the foundation for Google's web search tools. Langville and Meyer provide additional guidance on the construction and components of PageRank in their 2004 paper "Deeper Inside PageRank". Professor Mike Izbicki provided valuable guidance throughout this project.

We begin by constructing a Markov matrix P, where each entry ij "is the proabability of moving from state i to state j" (Langville and Meyer 2004). This matrix is transformed into a stochastic, irreducible, and primitive matrix. This Markov matrix will converge to the dominant eigenvector. This vector is the PageRank vector which indicates the importance of each webpage within a graph. To do so, Brin and Page use the power method, which stores just the previous iterate for each iteration, and converges quickly for the stochastic, irreducible, and primitive P (P bar bar) matrix.

### Task 1: Implementing the Power Method
**Part 1: Basic Power Method Implementation**
After implementing the algorithm, we can test the basic algorithm on a 6 webpage graph, similar to the one used in *Langville and Meyer 2004*. We obtain the following result:

The verbose tag has been turned on to demonstrate how in the DEBUG statements, the algorithm converges, toward the constant *epsilon* term 1e-6. The pages have been ranked with the fourth URL being the highest ranked, and the first URL being the lowest. 

```
run pagerank.py --data=small.csv.gz --verbose

DEBUG:root:computing indices
DEBUG:root:computing values
INFO:root:rank=0 pagerank=6.0257e-01 url=4
INFO:root:rank=1 pagerank=4.6414e-01 url=6
INFO:root:rank=2 pagerank=3.4544e-01 url=5
INFO:root:rank=3 pagerank=1.9498e-01 url=2
INFO:root:rank=4 pagerank=9.9210e-02 url=3
INFO:root:rank=5 pagerank=8.9347e-02 url=1
```

**Part 2: Search Queries**
We can use a number of command line arguments to refine our search. The algorithm will now return those pages most relevant to our queries. The <code>--search_query</code> argument accepts a string and compares it with each links and filters out those links that do not include the query. For this portion, I use a dataset prepared by Professor Mike Izbicki that graphs hyperlinks from the defense blog www.lawfareblog.com. From this point forward, I will not use the verbose command to make results more concise. 

If we make our search query 'corona' to find articles about the pandemic, the following links are the most relevant according to PageRank:

```
run pagerank.py --data=lawfareblog.csv.gz --search_query=corona

INFO:root:rank=0 pagerank=4.5861e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=4.0460e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=2.6116e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=2.5390e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=2.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=2.2895e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=2.2727e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=7 pagerank=2.2520e-03 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=8 pagerank=2.1878e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=9 pagerank=2.0339e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus

```

Now running the algorithm using the search query 'trump':
```
run pagerank.py --data=lawfareblog.csv.gz --search_query=trump

INFO:root:rank=0 pagerank=6.6243e-02 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=1 pagerank=6.0194e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns   
INFO:root:rank=2 pagerank=3.4969e-02 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements      
INFO:root:rank=3 pagerank=3.2193e-02 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=4 pagerank=3.0971e-02 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=5 pagerank=2.8460e-02 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=2.5252e-02 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=2.2457e-02 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=2.1462e-02 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=2.1103e-02 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors
```

Changing the search query to other national security issues, for example, 'iran':
```
run pagerank.py --data=lawfareblog.csv.gz --search_query=iran

INFO:root:rank=0 pagerank=6.6131e-02 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=2.9199e-02 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=1.7709e-02 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.4604e-02 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge        
INFO:root:rank=4 pagerank=8.4512e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=5 pagerank=8.3989e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=6 pagerank=8.2581e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=8.0561e-03 url=www.lawfareblog.com/trump-moves-cut-irans-oil-revenues-whats-his-endgame
INFO:root:rank=8 pagerank=7.1939e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=9 pagerank=5.9405e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
```

**Part 3: Concerns about the structure of webpages**
Most websites have a lot of structure, as most pages are connected to the homepage and some other broad pages, like www.lawfareblog.com/topics. Because PageRank does link ranking, those sites that many pages link to will often, but not always, have a higher rating. If we examine the largest PageRanks across www.lawfareblog.com, we can see a several of these broad pages appear, by running the code:

```
run pagerank.py --data=lawfareblog.csv.gz

INFO:root:rank=0 pagerank=8.4156e+00 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=1 pagerank=8.4156e+00 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=8.4156e+00 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=3 pagerank=8.4156e+00 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=4 pagerank=8.4156e+00 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=5 pagerank=8.4156e+00 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=6 pagerank=8.4156e+00 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=8.4156e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=8 pagerank=8.4156e+00 url=www.lawfareblog.com/topics
INFO:root:rank=9 pagerank=8.4156e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
```

However, these pages are not very interesting because they are not articles. How can we find the most important articles? We'd have to modify the P matrix by removing all links to non-article pages.

How do we know if a link is a non-article page? This is a hard question to answer with 100% accuracy, but there are many methods that get us most of the way there. An easy method is to remove all pages that have "too many" links. The <code>--filter_ratio</code> argument removes all pages that have more links than the specified fraction. 

Using this option, we can estimate the most important articles on the domain with the following command:

```
run pagerank.py --data=lawfareblog.csv.gz --filter_ratio=0.2

INFO:root:rank=0 pagerank=4.2773e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.7717e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.7533e+00 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.8720e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
INFO:root:rank=4 pagerank=1.7417e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
INFO:root:rank=5 pagerank=1.7411e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
INFO:root:rank=6 pagerank=1.7347e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
INFO:root:rank=7 pagerank=1.6384e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
INFO:root:rank=8 pagerank=1.5597e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
INFO:root:rank=9 pagerank=9.1265e-01 url=www.lawfareblog.com/events
```

The <code>--filter_ratio</code> essentially acts sort of an "anti-spam" filter.

**Part 4: Eigengaps**
The eigengap of the P barbar matrix is bounded by the alpha parameter. The size of the eigengap determines determines the speed at which the algorithm converges. If the eigengap of a matrix is large, then it will converge quickly even at alpha values that approach 1. If the eigengap is small, then only at smaller alpha values will the convergence occur quickly. We can run the following four commands to illustrate this:

```
run pagerank.py --data=lawfareblog.csv.gz --verbose

INFO:root:rank=0 pagerank=8.4156e+00 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=1 pagerank=8.4156e+00 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=8.4156e+00 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=3 pagerank=8.4156e+00 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=4 pagerank=8.4156e+00 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=5 pagerank=8.4156e+00 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=6 pagerank=8.4156e+00 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=8.4156e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=8 pagerank=8.4156e+00 url=www.lawfareblog.com/topics
INFO:root:rank=9 pagerank=8.4156e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
```

```
run pagerank.py --data=lawfareblog.csv.gz --verbose --alpha=0.999999

INFO:root:rank=0 pagerank=1.0703e+01 url=www.lawfareblog.com/topics
INFO:root:rank=1 pagerank=1.0703e+01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=1.0703e+01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=3 pagerank=1.0703e+01 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=4 pagerank=1.0703e+01 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=5 pagerank=1.0703e+01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=6 pagerank=1.0703e+01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=7 pagerank=1.0703e+01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=8 pagerank=1.0703e+01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=9 pagerank=1.0703e+01 url=www.lawfareblog.com/documents-related-mueller-investigation
```

```
run pagerank.py --data=lawfareblog.csv.gz --verbose --filter_ratio=0.2

INFO:root:rank=0 pagerank=4.2773e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.7717e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.7533e+00 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.8720e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
INFO:root:rank=4 pagerank=1.7417e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
INFO:root:rank=5 pagerank=1.7411e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
INFO:root:rank=6 pagerank=1.7347e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
INFO:root:rank=7 pagerank=1.6384e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
INFO:root:rank=8 pagerank=1.5597e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
INFO:root:rank=9 pagerank=9.1265e-01 url=www.lawfareblog.com/events
```

```
run pagerank.py --data=lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.9999999

INFO:root:rank=0 pagerank=4.7991e+01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=4.7991e+01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=7.2782e+00 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=2.1711e+00 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=1.4217e+00 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.0878e+00 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.0878e+00 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.0878e+00 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=8 pagerank=1.0878e+00 url=www.lawfareblog.com/water-wars-us-china-divide-shangri-la
INFO:root:rank=9 pagerank=1.0878e+00 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
```

### Task 2: The Personalization Vector
**Part 1: Filterting with a personalization vector**

Using personalization vector to filter the personalization vector Webpage is considered important if other coronavirus websites think that this website is important. Comparing the results we get from using <code>--personalization_vector_query</code> and the results from <code>--search_query</code> to illustrate:

```
run pagerank.py --data=lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query=corona --search_query=-corona

INFO:root:rank=0 pagerank=8.8870e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=8.8867e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.8256e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.0729e-01 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=9.4298e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=5 pagerank=7.9633e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=6 pagerank=7.5307e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=6.8115e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=6.4847e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm        
INFO:root:rank=9 pagerank=6.4847e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```

```
run pagerank.py --data=lawfareblog.csv.gz --filter_ratio=0.2 --search_query=corona

INFO:root:rank=0 pagerank=1.1602e-01 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=1 pagerank=5.6374e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=5.0830e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=5.0481e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=4.8031e-02 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=5 pagerank=4.7743e-02 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=4.3727e-02 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=7 pagerank=2.5817e-02 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=2.5463e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=1.9066e-02 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
```

With the <code>--personalization_vector_query</code> option, a webpage is important only if other coronavirus webpages also think it's important; with the <code>--search_query</code> option, a webpage is important if any other webpage thinks it's important. Notice that many of the webpages are about Congressional proceedings related to the coronavirus. From a strictly coronavirus perspective, these are not very important webpages. But in the broader context of national security, these are very important webpages.


**Part 2: Similar topics**
Another use of the <code>--personalization_vector_query</code> option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus. This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their corona importance, but removes webpages mentioning corona from the results:

```
run pagerank.py --data=lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query=corona --search_query=-corona

INFO:root:rank=0 pagerank=8.8870e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=8.8867e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.8256e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.0729e-01 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=9.4298e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=5 pagerank=7.9633e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=6 pagerank=7.5307e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=6.8115e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=6.4847e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm        
INFO:root:rank=9 pagerank=6.4847e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america

```

Notice that several url results talk about concepts that are obviously related to "coronavirus" like "covid", "clinical trials", and "quarantine". But this algorithm also finds articles about Chinese propaganda and Trump's policy decisions. Both of these articles are highly relevant to coronavirus discussions, but a simple keyword search for corona or related terms would not find these articles.


**Part 3: Other National Security Topics**
Using the issue about Iran as an example to conduct a similar experiment as the one above (in task 2 part 2):
```
run pagerank.py --data=lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query=iran --search_query=-iran

INFO:root:rank=0 pagerank=4.5063e-01 url=www.lawfareblog.com/omphalos
INFO:root:rank=1 pagerank=2.5712e-01 url=www.lawfareblog.com/cancellation-algerias-elections-opportunity-democratization        
INFO:root:rank=2 pagerank=2.5394e-01 url=www.lawfareblog.com/yemen-houthi-strategy-has-promise-and-risk
INFO:root:rank=3 pagerank=2.5307e-01 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=4 pagerank=2.5307e-01 url=www.lawfareblog.com/haftar-attacking-tripoli-us-needs-re-engage-libya
INFO:root:rank=5 pagerank=2.0710e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns   
INFO:root:rank=6 pagerank=1.9135e-01 url=www.lawfareblog.com/blurred-distinction-between-armed-conflict-and-civil-unrest-recent-events-gaza
INFO:root:rank=7 pagerank=1.8959e-01 url=www.lawfareblog.com/document-sen-tim-kaine-presses-pentagon-legal-definition-collective-self-defense
INFO:root:rank=8 pagerank=1.8959e-01 url=www.lawfareblog.com/document-july-2018-nato-summit-communique
INFO:root:rank=9 pagerank=1.8942e-01 url=www.lawfareblog.com/al-kibar-strike-what-difference-26-years-make
```

```
run pagerank.py --data=lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query=tiktok --search_query=-tiktok

INFO:root:rank=0 pagerank=8.9733e-02 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=8.9728e-02 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=4.4576e-02 url=www.lawfareblog.com/senate-examines-threats-homeland
INFO:root:rank=3 pagerank=4.4176e-02 url=www.lawfareblog.com/new-reporting-offers-glimpse-inside-chinas-xinjiang-practices-renewing-calls-us-response
INFO:root:rank=4 pagerank=3.5131e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns   
INFO:root:rank=5 pagerank=3.4980e-02 url=www.lawfareblog.com/announcing-huawei-5g-and-national-security-lawfare-compilation-new-lawfare-e-book
INFO:root:rank=6 pagerank=2.7952e-02 url=www.lawfareblog.com/us-china-resume-trade-talks-obstacles-remain
INFO:root:rank=7 pagerank=2.6504e-02 url=www.lawfareblog.com/trump-encourages-china-investigate-biden-family-clouding-trade-talks
INFO:root:rank=8 pagerank=2.6504e-02 url=www.lawfareblog.com/us-china-move-toward-october-trade-talks
INFO:root:rank=9 pagerank=2.5292e-02 url=www.lawfareblog.com/rethinking-encryption
```
