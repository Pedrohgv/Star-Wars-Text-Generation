from bs4 import BeautifulSoup as bs
import requests
import lxml
import time
import pandas as pd
from progressbar import progressbar as pb

from requests.exceptions import ConnectionError, MissingSchema

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36'}


class Article:
    """Class that defines an article. It contains a title, a list of categories and a text
    """

    class InvalidArticle(Exception):
        """Exception that is raised if any attributes can't be found
        """

        def __init__(self, error_message):
            """ Sets error message according to string passed
            """
            if error_message:
                self.message = error_message
            else:
                self.message = 'Unspecified article error'

        def __str__(self):
            return self.message

    # articles are to be rejected if they belong to any of these categories
    reject = ['Real-world articles',
              'Star Wars media by canonicity',
              'Legends articles',
              'Pages with missing permanent archival links',
              'Disambiguation pages',
              'Years']

    subscripts = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]', '[10]',
                  '[11]', '[12]', '[13]', '[14]', '[15]', '[16]', '[17]', '[18]', '[19]', '[20]']

    def __init__(self, URL=None, verbosity=True, attempts=3):

        if verbosity:
            print('Acquiring article...')

        soup = self.get_random_article(URL, attempts)  # gets article

        try:
            self.title = self.get_article_title(soup)
            self.categories = self.get_article_categories(soup)
            self.text = self.get_info(soup)
        except:
            raise

        if verbosity:
            print('Article Acquired!')

    def get_random_article(self, URL=None, attempts=3):
        """Get Beautiful Soup object from URL

        Keyword Arguments:
            URL {URL} -- If URL is defined, then the function will return the specified article
            attempts {int} -- Max number of attempts to get the pag

        Returns:
            beautifulsoup -- A beautifulSoup Object
        """

        if not URL:
            # URL that gets a random page
            URL = 'https://starwars.fandom.com/wiki/Special:Random'

        i = 0
        while True:
            # tries to get the page for defined number of tries
            try:
                time.sleep(0.01)
                page = requests.get(URL, headers=headers)
                break
            except ConnectionError:     # if connection error occurs, try again
                time.sleep(1)
                pass
            except:         # general exception, try 3 times
                i += 1
                if i >= attempts:
                    raise
                time.sleep(0.01)
                pass

        soup = bs(page.content, 'lxml')

        return soup

    def get_article_title(self, soup):
        """Gets the title from an article

        Arguments:
            soup {BeautifulSoup Object} -- BeautifulSoup Object

        Returns:
            [title] -- Article's title
        """
        # checks if title can be found
        try:
            title = soup.find('h1', class_='page-header__title').text
        except:
            raise self.InvalidArticle('Cannot find title')

        return title

    def get_article_categories(self, soup):
        """Gets categories to witch the article belongs

        Arguments:
            soup {BeautifulSoup Object} -- BeautifulSoup Object to be analyzed

        Returns:
            list -- The categories to which the article belongs
        """
        # Checks if categories can be found
        try:
            categories = soup.find(
                'div', class_='page-header__categories-links')
            categories = categories.find_all('a', recursive=False)
            categories = [tag.text for tag in categories]
        except:
            raise self.InvalidArticle('Cannot find categories')

        return categories

    def get_info(self, soup):
        """Gets the first information from an article

        Arguments:
            soup {BeautifulSoup Object} -- BeautifulSoup object to be analyzed

        Returns:
            string -- Header of article
        """
        try:
            # isolates content from page
            soup = soup.find('div', id='mw-content-text')
            # isolates the article's brief description
            text = soup.find('p', recursive=False).text
            # remove reference subscripts from text
            for subscript in self.subscripts:
                text = text.replace(subscript, '')
        except:
            raise self.InvalidArticle('Cannot find content')

        return text

    def is_suited(self):
        """Returns either if the article is suited or not

        Returns:
            bool -- True if is suited, False otherwise
        """
        return not any(category in self.reject for category in self.categories)


def create_random_database(size=1000, name='Articles.csv', attempts=3):
    """Creates a database of articles and saves it on a CSV file

    Keyword Arguments:
        size {int} -- Number of articles (default: {1000})
        name {string} -- Name of csv file to be created
        attempts {int} -- Max number of attempts to get the page

    Returns:
        [type] -- [description]
    """

    print('Scraping data...')
    time.sleep(0.5)

    data = []
    for _ in pb(range(size)):

        while True:
            article = Article(verbosity=False, attempts=attempts)

            # checks if article belongs to any of the 'forbidden' categories
            if article.is_suited():
                data.append(
                    [article.title, article.text, article.categories])
                break

    print('Creating CSV File...')
    df = pd.DataFrame(data, columns=['Title', 'Text', 'Categories'])
    df.to_csv(name)

    print('Done!')

    return None


def create_complete_database(max_num_pages=None, name='Complete Database.csv', attempts=3):
    """Scraps all canon and valid articles from wookiepedia and saves them on a file
    """

    # initial listing page
    listing_URL = 'https://starwars.fandom.com/wiki/Category:Canon_articles'

    # root URL to be added to scraped article links
    base_URL = 'https://starwars.fandom.com'

    links = []  # list that contains all the links to the articles

    page = 1
    while True:
        # gets the listing page
        listing_soup = requests.get(listing_URL, headers)
        listing_soup = bs(listing_soup.content, 'lxml')

        # append links found on the current listing page
        print('\rGetting links from page number {}...'.format(page), end='')

        links.extend([base_URL + link['href']
                      for link in listing_soup.find_all('a', class_='category-page__member-link')])

        try:    # checks if there are more listing pages
            listing_URL = listing_soup.find(
                'a', class_='category-page__pagination-next')['href']  # URL of the next listing page
            if max_num_pages:
                if page >= max_num_pages:
                    break
            page += 1
        except TypeError:
            break
    print('\rLink scraping complete: {} links found from {} listing pages'.format(
        len(links), page))

    print('Scraping data...')
    time.sleep(0.5)

    # save links on a .txt file
    links_file = open('Wookiepedia Links.txt', 'w')
    for link in links:
        links_file.write(link + '\n')

    data = []
    for link in pb(links):
        try:
            article = Article(URL=link, verbosity=False, attempts=attempts)

            # checks if article belongs to any of the 'forbidden' categories
            if article.is_suited():
                data.append(
                    [article.title, article.text, article.categories])
        except MissingSchema:   # if link is broken, just don't append to data
            pass
        except:
            pass

    print('Creating CSV File...')
    df = pd.DataFrame(data, columns=['Title', 'Text', 'Categories'])
    df.to_csv(name)

    print('Done!')

    return None
