import os
import whois
import math
import re
import socket
import Levenshtein
import tldextract
import requests
import concurrent.futures
import dns.resolver
import time
import logging
import subprocess
import json

from functools import wraps
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, urljoin
from urllib.request import urlopen
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def timer(func):
    """Record execution time of any functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        if args and hasattr(args[0], "exec_time"):
            args[0].exec_time += elapsed_time
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds, cumulative exec_time: {args[0].exec_time:.2f} seconds"
            )
        else:
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds (no instance with exec_time found)"
            )
        return result  # Return the original function's result

    return wrapper


except_funcs = [
    "get_state_and_page",
    "global_rank",
    "page_rank",
    "google_index",
    "dns_record",
    "count_internal_redirect",
    "count_external_redirect",
    "count_internal_error",
    "count_external_error",
]


def deadline(timeout):
    """Deadline execution time of any functions"""

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    if func.__name__ == except_funcs[0]:  # get_state_and_page
                        return (0, None)
                    elif func.__name__ in except_funcs[1:3]:  # global_rank, page_rank
                        return -1
                    elif func.__name__ in except_funcs[3:5]:  # google_index, dns_record
                        return 1
                    else:  # count_internal_redirect, count_external_redirect, count_internal_error, count_external_error
                        return 0

        return wrapper

    return decorate


def wrap_value(value):
    if callable(value):
        return value
    return lambda: value


class URL_EXTRACTOR(object):
    # Cache
    global_rank_cache = {}
    page_rank_cache = {}
    whois_cache = {}
    dns_record_cache = {}

    @timer
    def __init__(self, url, label="Unknown", enable_logging=False):
        # Logging & thời gian
        self.exec_time = 0.0
        self.log_level = logging.INFO if enable_logging else logging.WARNING
        logging.getLogger().setLevel(self.log_level)

        # Các giá trị mặc định để tránh AttributeError
        self.url = url
        self.label = label
        self.hostname = ""
        self.domain = ""
        self.subdomain = ""
        self.tld = ""
        self.path = ""
        self.query = ""
        self.scheme = ""
        self.words_raw = []
        self.words_raw_host = []
        self.words_raw_path = []
        self.hints = [
            "wp","login","includes","admin","content","site","images",
            "js","alibaba","css","myaccount","dropbox","themes",
            "plugins","signin","view"
        ]
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        self.allbrands = []
        self.res = None

        # 4. Parse URL
        try:
            self.p = urlparse(self.url)
        except Exception:
            class _P: pass
            self.p = _P()
            self.p.hostname = ""
            self.p.path = ""
            self.p.query = ""
            self.p.scheme = ""

        try:
            self.extracted = tldextract.extract(self.url)
        except Exception:
            class _E: pass
            self.extracted = _E()
            self.extracted.domain = ""
            self.extracted.suffix = ""
            self.extracted.subdomain = ""

        self.hostname = getattr(self.p, "hostname", "") or ""
        if getattr(self.extracted, "domain", ""):
            if getattr(self.extracted, "suffix", ""):
                self.domain = f"{self.extracted.domain}.{self.extracted.suffix}"
            else:
                self.domain = self.extracted.domain
        else:
            self.domain = self.hostname or ""
        self.subdomain = getattr(self.extracted, "subdomain", "") or ""
        self.tld = getattr(self.extracted, "suffix", "") or ""
        self.path = getattr(self.p, "path", "") or ""
        self.query = getattr(self.p, "query", "") or ""
        self.scheme = getattr(self.p, "scheme", "") or ""

        # 5. Word extraction
        try:
            self.words_raw, self.words_raw_host, self.words_raw_path = self.words_raw_extraction()
        except Exception as e:
            logging.warning("words_raw_extraction failed: %s", e)
            self.words_raw, self.words_raw_host, self.words_raw_path = ([], [], [])

        # 7. Allbrands
        try:
            self.allbrands_path = open(os.path.join(BASE_DIR, "scripts", "allbrands.txt"), "r")
            self.allbrands = self.__txt_to_list()
        except Exception as e:
            logging.warning("could not load allbrands.txt: %s", e)
            self.allbrands = []

        # 8. Sus TLD + API key
        self.suspecious_tlds = [
            "fit","tk","gp","ga","work","ml","date","wang","men","icu",
            "online","click","country","stream","download","xin","racing","jetzt",
            "ren","mom","party","review","trade","accountants","science","ninja","xyz",
            "faith","zip","cricket","win","accountant","realtor","top","christmas","gdn",
            "link","asia","club","la","ae","exposed","pe","go.id","rs","k12.pa.us","or.kr",
            "ce.ke","audio","gob.pe","gov.az","website","bj","mx","media","sa.gov.au"
        ]
        self.OPR_API_key = "gk8cg0gckckwk8gso88ss4c888cs4csc480s00o8"

    @timer
    @deadline(5)
    def get_whois(self):
        if self.domain in URL_EXTRACTOR.whois_cache:
            return URL_EXTRACTOR.whois_cache[self.domain]
        try:
            result = whois.whois(self.domain)
            URL_EXTRACTOR.whois_cache[self.domain] = result
            return result
        except Exception as e:
            URL_EXTRACTOR.whois_cache[self.domain] = None
            return None

    @timer
    def __txt_to_list(self):
        list = []
        for line in self.allbrands_path:
            list.append(line.strip())
        self.allbrands_path.close()
        return list

    @timer
    def words_raw_extraction(self):
        w_domain = re.split(
            r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", (self.domain or "").lower()
        )
        w_subdomain = re.split(
            r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", (self.subdomain or "").lower()
        )
        w_path = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", (self.path or "").lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        raw_words = list(filter(None, raw_words))
        return raw_words, list(filter(None, w_host)), list(filter(None, w_path))

    @staticmethod
    @timer
    def is_valid_url(url):
        try:
            result = re.match(r"^https?://[^\s/$.?#].[^\s]*$", url)
            parsed = urlparse(url)
            return bool(result and parsed.scheme and parsed.netloc)
        except:
            return False

    #######################################################################################
    #  ___     _   _ ____  _     _                                                        #
    # |_ _|   | | | |  _ \| |   ( )___                                                    #
    #  | |    | | | | |_) | |   |// __|                                                   #
    #  | | _  | |_| |  _ <| |___  \__ \                                                   #
    # |___(_)__\___/|_| \_\_____| |___/__  _____ ____                                     #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    #######################################################################################
    #                            1.1 Entropy of URL                                       #
    #######################################################################################

    @timer
    def entropy(self):
        str = self.url.strip()
        prob = [float(str.count(c)) / len(str) for c in dict.fromkeys(list(str))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    #######################################################################################
    #                           1.2 Having IP address in hostname                         #
    #######################################################################################

    @timer
    def having_ip_address(self):
        match = re.search(
            r"(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\."
            r"([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|"  # IPv4
            r"((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|"  # IPv4 in hexadecimal
            r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|"
            r"[0-9a-fA-F]{7}",
            self.url,
        )  # Ipv6
        if match:
            return 1
        else:
            return 0

    #######################################################################################
    #                     1.3 Total number of digits in URL string                        #
    #######################################################################################

    @timer
    def count_digits(self):
        return len(re.sub(r"[^0-9]", "", self.url))

    #######################################################################################
    #                    1.4 Total number of characters in URL string                     #
    #######################################################################################

    @timer
    def url_len(self):
        return len(self.url)

    @timer
    def hostname_len(self):
        return len(self.hostname)

    #######################################################################################
    #                    1.5 Total number of query parameters in URL                      #
    #######################################################################################

    @timer
    def count_parameters(self):
        params = self.url.split("&")
        return len(params) - 1

    #######################################################################################
    #                         1.6 Total Number of Fragments in URL                        #
    #######################################################################################

    @timer
    def count_fragments(self):
        fragments = self.url.split("#")
        return len(fragments) - 1

    #######################################################################################
    #                         1.7 URL shortening                                          #
    #######################################################################################

    @timer
    def has_shortening_service(self):
        match = re.search(
            r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
            r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
            r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
            r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|"
            r"db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|"
            r"q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
            r"x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
            r"tr\.im|link\.zip\.net",
            self.url,
        )
        if match:
            return 1
        else:
            return 0

    #######################################################################################
    #                         1.8 Count at (@) symbol at base url                        #
    #######################################################################################

    @timer
    def count_at(self):
        return self.url.count("@")

    #######################################################################################
    #                         1.9 Count comma (,) symbol at base url                    #
    #######################################################################################

    @timer
    def count_comma(self):
        return self.url.count(",")

    #######################################################################################
    #                         1.10 Count dollar ($) symbol at base url                    #
    #######################################################################################

    @timer
    def count_dollar(self):
        return self.url.count("$")

    #######################################################################################
    #                         1.11 Count semicolumn (;) symbol at base url                #
    #######################################################################################

    @timer
    def count_semicolumn(self):
        return self.url.count(";")

    #######################################################################################
    #                         1.12 Count (space, %20) symbol at base url                  #
    #######################################################################################

    @timer
    def count_space(self):
        return self.url.count(" ") + self.url.count("%20")

    #######################################################################################
    #                         1.13 Count and (&) symbol at base url                       #
    #######################################################################################

    @timer
    def count_and(self):
        return self.url.count("&")

    #######################################################################################
    #                         1.14 Count redirection (//) symbol at full url              #
    #######################################################################################

    @timer
    def count_double_slash(self):
        positions = [x.start(0) for x in re.finditer("//", self.url)]
        if positions and positions[-1] > 6:
            return 1
        else:
            return 0

    #######################################################################################
    #                         1.15 Count slash (/) symbol at full url                     #
    #######################################################################################

    @timer
    def count_slash(self):
        return self.url.count("/")

    #######################################################################################
    #                         1.16 Count equal (=) symbol at base url                     #
    #######################################################################################

    @timer
    def count_equal(self):
        return self.url.count("=")

    #######################################################################################
    #                         1.17 Count percentage (%) symbol at base url                #
    #######################################################################################

    @timer
    def count_percentage(self):
        return self.url.count("%")

    #######################################################################################
    #                         1.18 Count exclamation (?) symbol at base url               #
    #######################################################################################

    @timer
    def count_exclamation(self):
        return self.url.count("?")

    #######################################################################################
    #                         1.19 Count underscore (_) symbol at base url                #
    #######################################################################################

    @timer
    def count_underscore(self):
        return self.url.count("_")

    #######################################################################################
    #                         1.20 Count hyphens (-) symbol at base url                   #
    #######################################################################################

    @timer
    def count_hyphens(self):
        return self.url.count("-")

    #######################################################################################
    #                         1.21 Count number of dots in hostname                       #
    #######################################################################################

    @timer
    def count_dots(self):
        return self.url.count(".")

    #######################################################################################
    #                         1.22 Count number of colon (:) symbol at base url           #
    #######################################################################################

    @timer
    def count_colon(self):
        return self.url.count(":")

    #######################################################################################
    #                         1.23 Count number of stars (*) symbol at base url           #
    #######################################################################################

    @timer
    def count_star(self):
        return self.url.count("*")

    #######################################################################################
    #                         1.24 Count number of OR (|) symbol at base url              #
    #######################################################################################

    @timer
    def count_or(self):
        return self.url.count("|")

    #######################################################################################
    #                         1.25 Path entension != .txt/.exe                            #
    #######################################################################################

    @timer
    def has_path_txt_extension(self):
        if self.path.endswith(".txt"):
            return 1
        return 0

    @timer
    def has_path_exe_extension(self):
        if self.path.endswith(".exe"):
            return 1
        return 0

    #######################################################################################
    #                         1.26 Count number of http or https in url path              #
    #######################################################################################

    @timer
    def count_http_token(self):
        combined = self.path + (("?" + self.query) if self.query else "")
        count = len(re.findall(r"https?", combined))
        return count

    #######################################################################################
    #                         1.27 Uses https protocol                                    #
    #######################################################################################

    @timer
    def has_https(self):
        if self.scheme == "https":
            return 0
        return 1

    #######################################################################################
    #                   1.28 Checks if tilde symbol exist in webpage URL                  #
    #######################################################################################

    @timer
    def count_tilde(self):
        if self.url.count("~") > 0:
            return 1
        return 0

    #######################################################################################
    #                   1.29 Number of phish-hints in url path                            #
    #######################################################################################

    @timer
    def count_phish_hints(self):
        count = 0
        for hint in self.hints:
            count += self.path.lower().count(hint)
        return count

    #######################################################################################
    #                   1.30 Check if TLD exists in the path                              #
    #######################################################################################

    @timer
    def has_tld_in_path(self):
        if self.path.lower().count(self.tld) > 0:
            return 1
        return 0

    #######################################################################################
    #                   1.31 Check if TLD exists in the path                              #
    #######################################################################################

    @timer
    def has_tld_in_subdomain(self):
        if self.subdomain.count(self.tld) > 0:
            return 1
        return 0

    #######################################################################################
    #                   1.32 Check if TLD in bad position                                 #
    #######################################################################################

    @timer
    def tld_in_bad_position(self):
        if (
            self.tld_in_path(self.tld, self.path) == 1
            or self.tld_in_subdomain(self.tld, self.subdomain) == 1
        ):
            return 1
        return 0

    #######################################################################################
    #                  1.33 Abnormal subdomain starting with wwww-, wwNN                  #
    #######################################################################################

    @timer
    def has_abnormal_subdomain(self):
        if re.search(r"(http[s]?://(w[w]?|\d))([w]?(\d|-))", self.url):
            return 1
        return 0

    #######################################################################################
    #                           1.34 Number of redirection                                #
    #######################################################################################

    @timer
    def count_redirection(self):
        try:
            return len(self.page.history)
        except AttributeError:
            return -1

    #######################################################################################
    #                   1.35 Number of redirection to different domains                   #
    #######################################################################################

    @timer
    def count_external_redirection(self):
        count = 0
        try:
            if len(self.page.history) == 0:
                return 0
            else:
                for i, response in enumerate(self.page.history, 1):
                    if self.domain.lower() not in response.url.lower():
                        count += 1
                    return count
        except AttributeError:
            return -1

    #######################################################################################
    #                           1.36 Consecutive Character Repeat                         #
    #######################################################################################

    @timer
    def char_repeat(self):
        def __all_same(items):
            return all(x == items[0] for x in items)

        repeat = {"2": 0, "3": 0, "4": 0, "5": 0}
        part = [2, 3, 4, 5]

        for word in self.words_raw:
            for char_repeat_count in part:
                for i in range(len(word) - char_repeat_count + 1):
                    sub_word = word[i : i + char_repeat_count]
                    if __all_same(sub_word):
                        repeat[str(char_repeat_count)] = (
                            repeat[str(char_repeat_count)] + 1
                        )
        return sum(list(repeat.values()))

    #######################################################################################
    #                              1.37 Puny code in domain                               #
    #######################################################################################

    @timer
    def has_punycode(self):
        if self.url.startswith("http://xn--") or self.url.startswith("http://xn--"):
            return 1
        else:
            return 0

    #######################################################################################
    #                              1.38 Domain in brand list                              #
    #######################################################################################

    @timer
    def has_domain_in_brand(self):
        if self.words_raw_host[0] in self.allbrands:
            return 1
        else:
            return 0

    @timer
    def has_domain_in_brand1(self):
        for d in self.allbrands:
            if len(Levenshtein.editops(self.words_raw_host[0].lower(), d.lower())) < 2:
                return 1
        return 0

    #######################################################################################
    #                              1.39 Brand name in path/domain                         #
    #######################################################################################

    @timer
    def has_brand_in_path(self):
        for b in self.allbrands:
            if "." + b + "." in self.path and b not in self.domain:
                return 1
        return 0

    @timer
    def has_brand_in_subdomain(self):
        subdomain_components = self.subdomain.split(".") if self.subdomain else []
        for b in self.allbrands:
            if b in subdomain_components and b not in self.domain:
                return 1
        return 0

    #######################################################################################
    #                              1.40 Count www in url words                            #
    #######################################################################################

    @timer
    def count_www(self):
        count = 0
        for word in self.words_raw:
            if not word.find("www") == -1:
                count += 1
        return count

    #######################################################################################
    #                              1.41 Count com in url words                            #
    #######################################################################################

    @timer
    def count_com(self):
        count = 0
        for word in self.words_raw:
            if not word.find("com") == -1:
                count += 1
        return count

    #######################################################################################
    #                          1.42 Check port presence in domain                         #
    #######################################################################################

    @timer
    def has_port(self):
        if re.search(
            r"^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)",
            self.url,
        ):
            return 1
        return 0

    #######################################################################################
    #                             1.43 Length of raw word list                            #
    #######################################################################################

    @timer
    def length_word_raw(self):
        return len(self.words_raw)

    #######################################################################################
    #                   1.44 Count average word length in raw word list                   #
    #######################################################################################

    @timer
    def average_word_raw_length(self):
        if len(self.words_raw) == 0:
            return 0
        return sum(len(word) for word in self.words_raw) / len(self.words_raw)

    @timer
    def average_word_raw_host_length(self):
        if len(self.words_raw_host) == 0:
            return 0
        return sum(len(word) for word in self.words_raw_host) / len(self.words_raw_host)

    @timer
    def average_word_raw_path_length(self):
        if len(self.words_raw_path) == 0:
            return 0
        return sum(len(word) for word in self.words_raw_path) / len(self.words_raw_path)

    #######################################################################################
    #                   1.45 longest word length in raw word list                         #
    #######################################################################################

    @timer
    def longest_word_raw_length(self):
        if len(self.words_raw) == 0:
            return 0
        return max(len(word) for word in self.words_raw)

    @timer
    def longest_word_raw_host_length(self):
        if len(self.words_raw_host) == 0:
            return 0
        return max(len(word) for word in self.words_raw_host)

    @timer
    def longest_word_raw_path_length(self):
        if len(self.words_raw_path) == 0:
            return 0
        return max(len(word) for word in self.words_raw_path)

    #######################################################################################
    #                   1.46 Shortest word length in word list                            #
    #######################################################################################

    @timer
    def shortest_word_raw_length(self):
        if len(self.words_raw) == 0:
            return 0
        return min(len(word) for word in self.words_raw)

    @timer
    def shortest_word_raw_host_length(self):
        if len(self.words_raw_host) == 0:
            return 0
        return min(len(word) for word in self.words_raw_host)

    @timer
    def shortest_word_raw_path_length(self):
        if len(self.words_raw_path) == 0:
            return 0
        return min(len(word) for word in self.words_raw_path)

    #######################################################################################
    #                              1.47 Prefix suffix                                     #
    #######################################################################################

    @timer
    def has_prefix_suffix(self):
        if re.findall(r"https?://[^\-]+-[^\-]+/", self.url):
            return 1
        else:
            return 0

    #######################################################################################
    #                              1.48 Count subdomain                                   #
    #######################################################################################

    @timer
    def count_subdomain(self):
        if len(re.findall(r"\.", self.url)) == 1:
            return 1
        elif len(re.findall(r"\.", self.url)) == 2:
            return 2
        else:
            return 3

    #######################################################################################
    #                             1.49 Statistical report                                 #
    #######################################################################################

    @timer
    def has_statistical_report(self):
        url_match = re.search(
            r"at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly",
            self.url,
        )
        try:
            ip_address = socket.gethostbyname(self.domain)
            ip_match = re.search(
                r"146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|"
                r"107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|"
                r"118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|"
                r"216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|"
                r"34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|"
                r"216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42",
                ip_address,
            )
            if url_match or ip_match:
                return 1
            elif ip_address == "125.235.4.59":
                return 2
            else:
                return 0
        except socket.gaierror:
            return 2
        except Exception:
            return 2

    #######################################################################################
    #                               1.50 Suspecious TLD                                   #
    #######################################################################################

    @timer
    def has_suspecious_tld(self):
        if self.tld in self.suspecious_tlds:
            return 1
        return 0

    #######################################################################################
    #                        1.51 Ratio of digits in url                                  #
    #######################################################################################

    @timer
    def ratio_digits_url(self):
        return len(re.sub(r"[^0-9]", "", self.url)) / len(self.url)

    #######################################################################################
    #                        1.52 Ratio of digits in hostname                             #
    #######################################################################################

    @timer
    def ratio_digits_hostname(self):
        if not self.hostname:
            return 0
        return len(re.sub(r"[^0-9]", "", self.hostname)) / len(self.hostname)

    #######################################################################################
    #  ___ ___ ___     _______  _______ _____ ____  _   _    _    _                       #
    # |_ _|_ _|_ _|   | ____\ \/ /_   _| ____|  _ \| \ | |  / \  | |                      #
    #  | | | | | |    |  _|  \  /  | | |  _| | |_) |  \| | / _ \ | |                      #
    #  | | | | | | _  | |___ /  \  | | | |___|  _ <| |\  |/ ___ \| |___                   #
    # |___|___|___(_) |_____/_/\_\ |_|_|_____|_| \_\_| \_/_/   \_\_____|                  #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    #######################################################################################
    #                             3.1 Datetime list converter                             #
    #######################################################################################

    @staticmethod
    @timer
    def normalize_datetime_list(date_list):
        """Convert a list of mixed naive and aware datetimes to a list of aware datetimes in UTC."""
        if not date_list:
            return []

        normalized_dates = []
        for dt in date_list:
            if dt is None:
                continue
            # Convert naive datetimes to aware ones (UTC)
            if dt.tzinfo is None:
                normalized_dates.append(dt.replace(tzinfo=timezone.utc))
            else:
                normalized_dates.append(dt.astimezone(timezone.utc))

        return normalized_dates

    #######################################################################################
    #                             3.2 Domain registration age                             #
    #######################################################################################

    @timer
    @deadline(5)
    def domain_registration_length(self):
        try:
            expiration_date = self.res.expiration_date

            # Handle case where expiration_date is a list
            if isinstance(expiration_date, list):
                # Normalize all dates to make them timezone-aware with UTC
                normalized_dates = self.normalize_datetime_list(expiration_date)
                if not normalized_dates:
                    return 0  # No valid data
                expiration_date = min(normalized_dates)
            elif expiration_date is None:
                return 0  # No data
            elif expiration_date.tzinfo is None:
                # Single naive datetime
                expiration_date = expiration_date.replace(tzinfo=timezone.utc)
            else:
                # Single aware datetime but potentially in a different timezone
                expiration_date = expiration_date.astimezone(timezone.utc)

            # Get current time in UTC
            now = datetime.now(timezone.utc)

            length_days = (expiration_date - now).days
            return length_days if length_days >= 0 else 0
        except Exception as e:
            logging.info(
                f"[WHOIS ERROR] domain_registration_length({self.domain}): {e}"
            )
            return -1

    #######################################################################################
    #                          3.3 Domain recognized by WHOIS                             #
    #######################################################################################

    @timer
    @deadline(5)
    def whois_registered_domain(self):
        try:
            hostname = self.res.domain_name
            if type(hostname) == list:
                for host in hostname:
                    if re.search(host.lower(), self.domain):
                        return 0
                return 1
            else:
                if re.search(hostname.lower(), self.domain):
                    return 0
                else:
                    return 1
        except:
            return 1

    #######################################################################################
    #                               3.4 Get web traffic                                   #
    #######################################################################################

    @timer
    def web_traffic(self):
        try:
            rank = BeautifulSoup(
                urlopen(
                    "http://data.alexa.com/data?cli=10&dat=s&url=" + self.url
                ).read(),
                "xml",
            ).find("REACH")["RANK"]
        except:
            return 0
        return int(rank)

    #######################################################################################
    #                                   3.5 Domain age                                    #
    #######################################################################################

    @timer
    @deadline(5)
    def domain_age(self):
        try:
            creation_date = self.res.creation_date

            # Handle case where creation_date is a list
            if isinstance(creation_date, list):
                # Normalize all dates to make them timezone-aware with UTC
                normalized_dates = self.normalize_datetime_list(creation_date)
                if not normalized_dates:
                    return -2  # No valid data
                creation_date = min(normalized_dates)
            elif creation_date is None:
                return -2  # No data
            elif creation_date.tzinfo is None:
                # Single naive datetime
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            else:
                # Single aware datetime but potentially in a different timezone
                creation_date = creation_date.astimezone(timezone.utc)

            # Get current time in UTC
            now = datetime.now(timezone.utc)

            age_days = (now - creation_date).days
            return age_days if age_days >= 0 else -2
        except Exception as e:
            logging.info(f"[WHOIS ERROR] domain_age({self.domain}): {e}")
            return -1

    #######################################################################################
    #                                  3.6 Global rank                                    #
    #######################################################################################

    @timer
    @deadline(3)
    def global_rank(self):
        if self.domain in URL_EXTRACTOR.global_rank_cache:
            return URL_EXTRACTOR.global_rank_cache[self.domain]

        try:
            rank_checker_response = requests.post(
                "https://www.checkpagerank.net/index.php",
                {"name": self.domain},
                timeout=5,  # Add timeout to prevent hanging
            )
            rank = int(
                re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0]
            )
        except (
            requests.exceptions.RequestException,
            ValueError,
            IndexError,
            Exception,
        ):
            # Return default value if network request fails
            rank = -1

        URL_EXTRACTOR.global_rank_cache[self.domain] = rank
        return rank

    #######################################################################################
    #                                 3.7 Google index                                    #
    #######################################################################################

    @timer
    @deadline(5)
    def google_index(self):
        param = {"q": "site:" + self.url}
        google_query = "https://www.google.com/search?" + urlencode(param)
        data = requests.get(google_query, headers=self.headers)
        soup = BeautifulSoup(data.text, "html.parser")
        try:
            if (
                "Our systems have detected unusual traffic from your computer network."
                in soup.text
            ):
                return -1
            # Look for search result links
            rso = soup.find(id="rso")
            if rso:
                links = rso.find_all("a", href=True)
                if links:
                    return 0  # Indexed
            return 1  # Not indexed
        except Exception:
            return 1  # Not indexed or error

    #######################################################################################
    #                        3.8 DNS record expiration length                             #
    #######################################################################################

    @timer
    @deadline(3)
    def dns_record(self):
        if self.domain in URL_EXTRACTOR.dns_record_cache:
            return URL_EXTRACTOR.dns_record_cache[self.domain]
        try:
            nameservers = dns.resolver.query(self.domain, "NS")
            if len(nameservers) > 0:
                result = 0
            else:
                result = 1
        except:
            result = 1
        URL_EXTRACTOR.dns_record_cache[self.domain] = result
        return result

    #######################################################################################
    #                           3.10 Page Rank from OPR                                   #
    #######################################################################################

    @timer
    @deadline(3)
    def page_rank(self):
        if self.domain in URL_EXTRACTOR.page_rank_cache:
            return URL_EXTRACTOR.page_rank_cache[self.domain]

        try:
            rank_checker_response = requests.post(
                "https://www.checkpagerank.net/index.php",
                {"name": self.domain},
                timeout=5,  # Add timeout to prevent hanging
            )
            rank = int(
                re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0]
            )
        except (
            requests.exceptions.RequestException,
            ValueError,
            IndexError,
            Exception,
        ):
            # Return default value if network request fails
            rank = -1

        URL_EXTRACTOR.page_rank_cache[self.domain] = rank
        return rank

    #######################################################################################
    #  _____     __   ____ ___  __  __ ____ ___ _   _ _____   URL's Features: 59          #
    # |_ _\ \   / /  / ___/ _ \|  \/  | __ )_ _| \ | | ____|  Content's Features: 27      #
    #  | | \ \ / /  | |  | | | | |\/| |  _ \| ||  \| |  _|    External Features: 6        #
    #  | |  \ V /   | |__| |_| | |  | | |_) | || |\  | |___   Total Features: 91          #
    # |___|_ \_(_)   \____\___/|_| _|_|____/___|_|_\_|_____|   (label included)           #
    # |  ___| ____|  / \|_   _| | | |  _ \| ____/ ___|                                    #
    # | |_  |  _|   / _ \ | | | | | | |_) |  _| \___ \                                    #
    # |  _| | |___ / ___ \| | | |_| |  _ <| |___ ___) |                                   #
    # |_|   |_____/_/   \_\_|  \___/|_| \_\_____|____/                                    #
    #######################################################################################

    @timer
    def extract_to_dataset(self):
        data = {}
        # List of (key, function) pairs for all features
        feature_funcs = [
            ("url", lambda: self.url),
            ("url_len", self.url_len),
            ("hostname_len", self.hostname_len),
            ("entropy", self.entropy),
            ("nb_fragments", self.count_fragments),
            ("nb_dots", self.count_dots),
            ("nb_hyphens", self.count_hyphens),
            ("nb_at", self.count_at),
            ("nb_exclamation", self.count_exclamation),
            ("nb_and", self.count_and),
            ("nb_or", self.count_or),
            ("nb_equal", self.count_equal),
            ("nb_underscore", self.count_underscore),
            ("nb_tilde", self.count_tilde),
            ("nb_percentage", self.count_percentage),
            ("nb_slash", self.count_slash),
            ("nb_dslash", self.count_double_slash),
            ("nb_star", self.count_star),
            ("nb_colon", self.count_colon),
            ("nb_comma", self.count_comma),
            ("nb_semicolumn", self.count_semicolumn),
            ("nb_dollar", self.count_dollar),
            ("nb_space", self.count_space),
            ("nb_http_token", self.count_http_token),
            ("nb_subdomain", self.count_subdomain),
            ("nb_www", self.count_www),
            ("nb_com", self.count_com),
            ("nb_redirection", self.count_redirection),
            ("nb_e_redirection", self.count_external_redirection),
            ("nb_phish_hints", self.count_phish_hints),
            ("has_ip", self.having_ip_address),
            ("has_https", self.has_https),
            ("has_punnycode", self.has_punycode),
            ("has_port", self.has_port),
            ("has_tld_in_path", self.has_tld_in_path),
            ("has_tld_in_subdomain", self.has_tld_in_subdomain),
            ("has_abnormal_subdomain", self.has_abnormal_subdomain),
            ("has_prefix_suffix", self.has_prefix_suffix),
            ("has_short_svc", self.has_shortening_service),
            ("has_path_txt_extension", self.has_path_txt_extension),
            ("has_path_exe_extension", self.has_path_exe_extension),
            ("has_domain_in_brand", self.has_domain_in_brand),
            ("has_brand_in_path", self.has_brand_in_path),
            ("has_sus_tld", self.has_suspecious_tld),
            ("has_statistical_report", self.has_statistical_report),
            ("word_raw_len", self.length_word_raw),
            ("char_repeat", self.char_repeat),
            ("shortest_word_raw_len", self.shortest_word_raw_length),
            ("shortest_word_raw_host_len", self.shortest_word_raw_host_length),
            ("shortest_word_raw_path_len", self.shortest_word_raw_path_length),
            ("longest_word_raw_len", self.longest_word_raw_length),
            ("longest_word_raw_host_len", self.longest_word_raw_host_length),
            ("longest_word_raw_path_len", self.longest_word_raw_path_length),
            ("avg_word_raw_len", self.average_word_raw_length),
            ("avg_word_raw_host_len", self.average_word_raw_host_length),
            ("avg_word_raw_path_len", self.average_word_raw_path_length),
            ("ratio_digits_url", self.ratio_digits_url),
            ("ratio_digits_host", self.ratio_digits_hostname),
            # Label
            ("label", lambda: self.label),
        ]

        for key, func in tqdm(
            feature_funcs, desc="  Extracting features", unit="feature"
        ):
            data[key] = wrap_value(func)()

        return data
