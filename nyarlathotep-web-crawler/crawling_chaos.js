const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require("fs");
const path = require("path");
const crypto = require('crypto');

puppeteer.use(StealthPlugin());

/**
 * Check if the page is valid and not an error/redirect page
 * @param {string} finalUrl - The final URL after redirects
 * @param {string} originalUrl - The original requested URL
 * @param {number} statusCode - HTTP status code
 * @param {string} title - Page title
 * @param {string} content - Page content
 * @returns {boolean} True if page is valid
 */
function isValidPage(finalUrl, originalUrl, statusCode, title, content) {
    // Rebuilt to only check status code <400, ignore other checks to accept minimal pages
    if (statusCode >= 400) {
        return false;
    }
    console.log('Debug isValidPage: Page is valid (status OK)');
    return true;
}

function normalizeUrl(inputUrl) {
    if (!inputUrl.startsWith('http://') && !inputUrl.startsWith('https://')) {
        return ['https://' + inputUrl, 'http://' + inputUrl];
    }
    return [inputUrl];
}

(async() => {
    const rawUrl = process.argv[2] || 'example.com';
    const candidateUrls = normalizeUrl(rawUrl);

    // detect môi trường
    const isColab = process.env.COLAB_RELEASE_TAG !== undefined || process.env.COLAB_GPU !== undefined;

    const browser = await puppeteer.launch({
        headless: isColab ? true : false, // Force visible browser để debug
        ignoreHTTPSErrors: true,
        args: [
            '--no-sandbox',
            '--no-first-run',
            '--disable-setuid-sandbox',
            '--disable-extensions',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-component-extensions-with-background-pages',
            '--disable-default-apps',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-client-side-phishing-detection',
            '--disable-popup-blocking',
            '--disable-blink-features=AutomationControlled',
            '--safebrowsing-disable-auto-update',
            '--safebrowsing-disable-download-protection',
        ]
    });

    const page = await browser.newPage();

    // Enable request interception to prevent following redirects
    await page.setRequestInterception(true);
    page.on('request', request => {
        if (request.isNavigationRequest() && request.redirectChain().length !== 0) {
            request.abort(); // Stop following redirect
        } else {
            request.continue();
        }
    });

    // Track redirects and responses
    let finalUrl = null;
    let redirectChain = [];
    let statusCode = null;
    let responseHeaders = {};

    // Listen to response events to track redirects and status codes
    page.on('response', response => {
        if (response.url() === page.url()) {
            statusCode = response.status();
            responseHeaders = response.headers();
            if (statusCode >= 300 && statusCode < 400) {
                console.log(`Detected redirect: ${statusCode}, Location: ${responseHeaders['location'] || 'unknown'}`);
            }
        }
    });

    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36');
    await page.setViewport({
        width: 1920,
        height: 1080
    });

    let targetUrl = null;
    let originalUrl = null;
    let is_alive = 0;

    for (const url of candidateUrls) {
        try {
            originalUrl = url;
            const response = await page.goto(url, {
                waitUntil: 'networkidle2', // Đổi sang networkidle2 để chờ load đầy đủ hơn
                timeout: 60000
            });

            finalUrl = page.url();
            statusCode = response.status();

            // Nếu là redirect, lấy content gốc
            let pageContent = await page.content();
            let pageTitle = await page.title();

            if (statusCode >= 300 && statusCode < 400) {
                console.log('Extracting from redirect page:', finalUrl);
            }

            // Check if page is actually accessible
            if (isValidPage(finalUrl, originalUrl, statusCode, pageTitle, pageContent)) {
                targetUrl = finalUrl;
                is_alive = 1;
                break;
            } else {
                console.error(`Invalid page detected for ${url}: redirected to ${finalUrl}, status: ${statusCode}`);
            }
        } catch (err) {
            console.error(`Failed with ${url}:`, err.message, err.stack);
        }
    }

    await new Promise(r => setTimeout(r, 10000)); // Tăng delay để JS load

    if (is_alive === 0) {
        const features = {
            original_url: originalUrl,
            final_url: finalUrl,
            is_alive: 0,
            redirect_detected: true,
            status_code: statusCode,
            extraction_timestamp: new Date().toISOString(),
            error_reason: 'Page not accessible or redirected to error page'
        };
        console.log(JSON.stringify(features));
        await browser.close();
        process.exit(0);
    }

    // Additional validation before feature extraction
    const pageValidation = await page.evaluate(() => {
        const title = document.title.toLowerCase();
        const bodyText = document.body ? document.body.textContent.toLowerCase() : '';

        // Check for common error page indicators
        const errorIndicators = [
            'page not found', '404', 'not found', 'error',
            'cloudflare', 'firebase hosting', 'github pages',
            'access denied', 'forbidden', 'unauthorized',
            'service unavailable', 'maintenance', 'coming soon',
            'domain for sale', 'parked domain', 'suspended'
        ];

        const hasErrorIndicators = errorIndicators.some(indicator =>
                title.includes(indicator) || bodyText.includes(indicator));

        // Check content length - error pages usually have minimal content
        const contentLength = bodyText.trim().length;
        const isMinimalContent = contentLength < 50; // Giảm ngưỡng

        // Check for generic page structures
        const hasGenericStructure = document.querySelectorAll('div').length < 3 &&
            document.querySelectorAll('p').length < 1; // Giảm ngưỡng

        console.log('Debug pageValidation:');
        console.log('- Has error indicators:', hasErrorIndicators);
        console.log('- Content length:', contentLength);
        console.log('- Is minimal content:', isMinimalContent);
        console.log('- Has generic structure:', hasGenericStructure);

        return {
            hasErrorIndicators,
            contentLength,
            isMinimalContent,
            hasGenericStructure,
            title: document.title,
            url: window.location.href
        };
    });

    // If validation fails, mark as invalid
    if (pageValidation.hasErrorIndicators) {
        const features = {
            original_url: originalUrl,
            final_url: finalUrl,
            is_alive: 0,
            redirect_detected: true,
            status_code: statusCode,
            validation_failed: true,
            validation_details: pageValidation,
            extraction_timestamp: new Date().toISOString(),
            error_reason: 'Page appears to be an error/placeholder page'
        };
        console.log(JSON.stringify(features));
        await browser.close();
        process.exit(0);
    }

    // Extract comprehensive content-based features using Puppeteer
    const features = await page.evaluate(() => {
        // Helper function to count elements safely
        const safeCount = (selector) => {
            try {
                return document.querySelectorAll(selector).length;
            } catch (e) {
                return 0;
            }
        };

        // Helper function to get text content safely
        const safeText = (element) => {
            try {
                return element ? element.textContent || '' : '';
            } catch (e) {
                return '';
            }
        };

        // Helper function to count special characters
        const countSpecialChars = (text) => {
            try {
                return text.split('').filter(c => !c.match(/[a-zA-Z0-9\s]/)).length;
            } catch (e) {
                return 0;
            }
        };

        // Helper function to count syllables in a word
        const countSyllables = (word) => {
            if (!word || typeof word !== 'string') {
                return 0;
            }

            // Convert to lowercase and remove non-alphabetic characters
            word = word.toLowerCase().replace(/[^a-z]/g, '');

            if (word.length === 0) {
                return 0;
            }

            // Handle special cases
            if (word.length <= 3) {
                return 1;
            }

            // Count vowel groups
            let syllableCount = 0;
            let previousWasVowel = false;
            const vowels = 'aeiouy';

            for (let i = 0; i < word.length; i++) {
                const isVowel = vowels.includes(word[i]);

                if (isVowel && !previousWasVowel) {
                    syllableCount++;
                }

                previousWasVowel = isVowel;
            }

            // Handle silent 'e' at the end
            if (word.endsWith('e') && syllableCount > 1) {
                syllableCount--;
            }

            // Handle special endings that add syllables
            const specialEndings = ['ed', 'es', 'er', 'le'];
            for (const ending of specialEndings) {
                if (word.endsWith(ending) && word.length > ending.length) {
                    const beforeEnding = word.slice(0, -ending.length);
                    if (ending === 'ed' && !vowels.includes(beforeEnding[beforeEnding.length - 1])) {
                        // Past tense 'ed' after consonant usually doesn't add syllable
                        continue;
                    } else if (ending === 'le' && !vowels.includes(beforeEnding[beforeEnding.length - 1])) {
                        // 'le' after consonant adds a syllable
                        syllableCount++;
                    }
                }
            }

            // Every word has at least one syllable
            return Math.max(1, syllableCount);
        };

        // Helper function to calculate Flesch Reading Ease Score
        const calculateFleschScore = (text) => {
            if (!text || typeof text !== 'string') {
                return 0;
            }

            // Clean the text
            const cleanText = text.trim().replace(/\s+/g, ' ');

            if (cleanText.length === 0) {
                return 0;
            }

            // Count sentences
            const sentences = cleanText
                .split(/[.!?]+\s+|[.!?]+$/)
                .filter(sentence => {
                    const trimmed = sentence.trim();
                    return trimmed.length >= 3 && /[a-zA-Z]/.test(trimmed);
                });

            if (sentences.length === 0) {
                return 0;
            }

            // Count words
            const words = cleanText
                .split(/\s+/)
                .filter(word => {
                    return word.length > 0 && /[a-zA-Z0-9]/.test(word);
                });

            if (words.length === 0) {
                return 0;
            }

            // Count syllables
            let totalSyllables = 0;
            words.forEach(word => {
                totalSyllables += countSyllables(word);
            });

            // Calculate metrics
            const averageSentenceLength = words.length / sentences.length;
            const averageSyllablesPerWord = totalSyllables / words.length;

            // Apply Flesch formula
            const fleschScore = 206.835 - (1.015 * averageSentenceLength) - (84.6 * averageSyllablesPerWord);

            // Round to 2 decimal places
            return Math.round(fleschScore * 100) / 100;
        };

        // Helper function to calculate average words per sentence
        const calculateAvgWordsPerSentence = (text) => {
            if (!text || typeof text !== 'string') {
                return 0;
            }

            // Clean the text: remove extra whitespace and normalize
            const cleanText = text.trim().replace(/\s+/g, ' ');

            if (cleanText.length === 0) {
                return 0;
            }

            // Split into sentences using multiple delimiters
            const sentences = cleanText
                .split(/[.!?]+\s+|[.!?]+$/)
                .filter(sentence => {
                    // Filter out empty sentences and very short fragments (< 3 characters)
                    const trimmed = sentence.trim();
                    return trimmed.length >= 3 && /[a-zA-Z]/.test(trimmed);
                });

            if (sentences.length === 0) {
                return 0;
            }

            // Count words in each sentence
            let totalWords = 0;
            sentences.forEach(sentence => {
                // Split by whitespace and count non-empty words
                const words = sentence.trim().split(/\s+/).filter(word => {
                    // Only count actual words (containing letters or numbers)
                    return word.length > 0 && /[a-zA-Z0-9]/.test(word);
                });
                totalWords += words.length;
            });

            return totalWords / sentences.length;
        };

        // Basic form and input features
        const nb_forms = safeCount('form');
        const nb_inputs = safeCount('input');
        const nb_buttons = safeCount('button');
        const nb_selects = safeCount('select');
        const nb_textareas = safeCount('textarea');

        // Media tag features
        const nb_imgs = safeCount('img');
        const nb_embeds = safeCount('embed');
        const nb_audios = safeCount('audio');
        const nb_videos = safeCount('video');

        // Script and link features
        const nb_scripts = safeCount('script');
        const nb_a = safeCount('a');
        const nb_links = safeCount('link');

        // Iframe features
        const nb_iframes = safeCount('iframe');

        // Content structure features
        const nb_divs = safeCount('div');
        const nb_spans = safeCount('span');
        const nb_paragraphs = safeCount('p');
        const nb_headings = safeCount('h1, h2, h3, h4, h5, h6');
        const nb_tables = safeCount('table');
        const nb_table_rows = safeCount('tr');
        const nb_table_cells = safeCount('td, th');
        const nb_ul_lists = safeCount('ul');
        const nb_ol_lists = safeCount('ol');
        const nb_list_items = safeCount('li');

        // Text emphasis features
        const nb_strongs = safeCount('strong');
        const nb_ems = safeCount('em');

        // Canvas and SVG features
        const nb_canvas = safeCount('canvas');
        const nb_svgs = safeCount('svg');

        // Meta and style features
        const nb_metas = safeCount('meta');
        const nb_styles = safeCount('style');

        // Text content features
        const bodyText = safeText(document.body);
        const inner_text_len = bodyText.length;
        const body_length = inner_text_len;

        // Script content length
        const scripts = document.querySelectorAll('script');
        let script_length = 0;
        scripts.forEach(script => {
            script_length += safeText(script).length;
        });

        // Title features
        const title = safeText(document.title);
        const is_empty_title = title ? 0 : 1;

        // Special characters count
        const nb_special_char = countSpecialChars(bodyText);

        // Additional content analysis
        const allElements = document.querySelectorAll('*');
        const totalElements = allElements.length;

        // Count elements with specific attributes
        let elements_with_onclick = 0;
        let elements_with_onmouse = 0;
        let elements_with_onload = 0;

        allElements.forEach(el => {
            if (el.onclick)
                elements_with_onclick++;
            if (el.onmouseover || el.onmouseout || el.onmousedown || el.onmouseup)
                elements_with_onmouse++;
            if (el.onload)
                elements_with_onload++;
        });

        // Check for specific content patterns
        const has_popup_window = bodyText.toLowerCase().includes('prompt(') ||
            bodyText.toLowerCase().includes('alert(') ||
            bodyText.toLowerCase().includes('confirm(');

        const has_right_click = bodyText.toLowerCase().includes('contextmenu') ||
            bodyText.toLowerCase().includes('event.button') ||
            bodyText.toLowerCase().includes('oncontextmenu');

        // Check for external resources
        const nb_external_links = Array.from(document.querySelectorAll('a[href^="http"]'))
            .filter(a => !a.href.includes(window.location.hostname)).length;

        const nb_external_scripts = Array.from(document.querySelectorAll('script[src^="http"]'))
            .filter(s => !s.src.includes(window.location.hostname)).length;

        const nb_external_images = Array.from(document.querySelectorAll('img[src^="http"]'))
            .filter(img => !img.src.includes(window.location.hostname)).length;

        // Text quality metrics
        const avg_sentence_len = calculateAvgWordsPerSentence(bodyText);
        const avg_words_per_sentence = calculateAvgWordsPerSentence(bodyText);
        const readability_score = calculateFleschScore(bodyText);

        // Content structure ratios
        const txt_to_html_ratio = bodyText.length / document.documentElement.outerHTML.length;
        const txt_to_tag_ratio = bodyText.length / totalElements;
        const link_density = nb_a / (bodyText.split(' ').length || 1);
        const img_density = nb_imgs / (bodyText.split(' ').length || 1);

        // Form complexity
        const nb_input_types = new Set(Array.from(document.querySelectorAll('input')).map(i => i.type)).size;
        const form_complexity = (nb_inputs + nb_selects + nb_textareas) / Math.max(nb_forms, 1);
        const has_password_field = document.querySelectorAll('input[type="password"]').length > 0 ? 1 : 0;
        const has_file_upload = document.querySelectorAll('input[type="file"]').length > 0 ? 1 : 0;

        // Security-related elements
        const has_ssl = bodyText.toLowerCase().includes('secure') ||
            bodyText.toLowerCase().includes('ssl') ||
            bodyText.toLowerCase().includes('encrypted') ? 1 : 0;

        const has_captcha = bodyText.toLowerCase().includes('captcha') ||
            document.querySelectorAll('[class*="captcha"], [id*="captcha"]').length > 0 ? 1 : 0;

        const has_login_form = document.querySelectorAll('input[type="password"]').length > 0 ? 1 : 0;

        // Date & time indicators
        const has_date_stamps = document.querySelectorAll('time, [datetime]').length;
        const has_new_badges = document.querySelectorAll('[class*="new"], [class*="badge"]').length;
        const copyright_year_now = new Date().getFullYear();
        const has_copyright_year = bodyText.includes(copyright_year_now.toString()) ? 1 : 0;

        // Social media presence
        const nb_social_media_links = Array.from(document.querySelectorAll('a[href]')).filter(a =>
                /facebook|twitter|instagram|linkedin|youtube|tiktok/.test(a.href.toLowerCase())).length;
        const has_social_sharing = document.querySelectorAll('[class*="share"], [class*="social"]').length > 0 ? 1 : 0;
        const nb_social_buttons = document.querySelectorAll('[class*="facebook"], [class*="twitter"], [class*="share"]').length;

        // Ad related content
        const has_ad_keywords = bodyText.toLowerCase().match(/(advertisement|sponsored|ad|banner)/g)?.length || 0;
        const has_sus_ad_class = document.querySelectorAll('[class*="ad"], [class*="banner"], [id*="ad"]').length;
        const has_popup_ad = bodyText.toLowerCase().includes('popup') ||
            bodyText.toLowerCase().includes('pop-up') ? 1 : 0;

        // User interaction element
        const has_comment_sections = document.querySelectorAll('[class*="comment"], [id*="comment"]').length > 0 ? 1 : 0;
        const has_rating_system = document.querySelectorAll('[class*="rating"], [class*="star"], [class*="review"]').length > 0 ? 1 : 0;
        const has_search_box = document.querySelectorAll('input[type="search"], [class*="search"]').length > 0 ? 1 : 0;

        // Mobile responsiveness
        const has_view_port_meta = document.querySelector('meta[name="viewport"]') ? 1 : 0;
        const has_responsive_imgs = document.querySelectorAll('img[srcset], picture').length;
        const has_mobile_css = Array.from(document.querySelectorAll('style, link[rel="stylesheet"]'))
            .some(el => el.textContent?.includes('@media') || el.href?.includes('mobile')) ? 1 : 0;

        // Advanced text analysis
        const unique_words_ratio = bodyText.length > 0 ? new Set(bodyText.toLowerCase().split(/\s+/)).size / bodyText.split(/\s+/).length : 0;
        const punctuation_density = (bodyText.match(/[.,;:!?]/g) || []).length / bodyText.length;
        const num_in_txt = (bodyText.match(/\d+/g) || []).length;
        const avg_paragraph_len = nb_paragraphs > 0 ? bodyText.length / nb_paragraphs : 0;

        return {
            // Basic form and input features
            nb_forms,
            nb_inputs,
            nb_buttons,
            nb_selects,
            nb_textareas,

            // Media tag features
            nb_imgs,
            nb_embeds,
            nb_audios,
            nb_videos,

            // Script and link features
            nb_scripts,
            nb_a,
            nb_links,

            // Iframe features
            nb_iframes,

            // Content structure features
            nb_divs,
            nb_spans,
            nb_paragraphs,
            nb_headings,
            nb_tables,
            nb_table_rows,
            nb_table_cells,
            nb_ul_lists,
            nb_ol_lists,
            nb_list_items,

            // Text emphasis features
            nb_strongs,
            nb_ems,

            // Canvas and SVG features
            nb_canvas,
            nb_svgs,

            // Meta and style features
            nb_metas,
            nb_styles,

            // Text content features
            inner_text_len,
            body_length,
            script_length,
            is_empty_title,
            nb_special_char,

            // Interactive elements
            elements_with_onclick,
            elements_with_onmouse,
            elements_with_onload,

            // Content patterns
            has_popup_window: has_popup_window ? 1 : 0,
            has_right_click: has_right_click ? 1 : 0,

            // External resources
            nb_external_links,
            nb_external_scripts,
            nb_external_images,

            // Text quality metrics
            avg_sentence_len,
            avg_words_per_sentence,
            readability_score,

            // Content structure ratios
            txt_to_html_ratio,
            txt_to_tag_ratio,
            link_density,
            img_density,

            // Form complexity
            nb_input_types,
            form_complexity,
            has_password_field,
            has_file_upload,

            // Security-related elements
            has_ssl,
            has_captcha,
            has_login_form,

            // Date & time indicators
            has_date_stamps,
            has_new_badges,
            has_copyright_year,

            // Social media presence
            nb_social_media_links,
            has_social_sharing,
            nb_social_buttons,

            // Ad related content
            has_ad_keywords,
            has_sus_ad_class,
            has_popup_ad,

            // User interaction element
            has_comment_sections,
            has_rating_system,
            has_search_box,

            // Mobile responsiveness
            has_view_port_meta,
            has_responsive_imgs,
            has_mobile_css,

            // Advanced text analysis
            unique_words_ratio,
            punctuation_density,
            num_in_txt,
            avg_paragraph_len,

            // Page statistics
            totalElements,
            title,
            url: window.location.href,
        };
    });

    if (!fs.existsSync("screenshots")) {
        fs.mkdirSync("screenshots");
    }

    // Use MD5 hash of the original URL to avoid long filenames
    const urlHash = crypto.createHash('md5').update(originalUrl).digest('hex');
    const screenshotPath = path.join("screenshots", `${urlHash}.png`);

    // Take screenshot with try-catch to handle errors like zero-width
    let screenshotTaken = false;
    try {
        // Check if page has content before screenshot (basic check)
        const contentLength = await page.evaluate(() => document.body.innerHTML.length);
        if (contentLength > 0) {
            await page.screenshot({
                path: screenshotPath,
                fullPage: true
            });
            screenshotTaken = true;
        } else {
            console.warn(`Skipping screenshot for ${targetUrl}: Page content is empty.`);
        }
    } catch (err) {
        console.error(`Screenshot failed for ${targetUrl}:`, err.message, err.stack);
    }

    // Add metadata including redirect information
    features.original_url = originalUrl;
    features.final_url = finalUrl;
    features.is_alive = is_alive;
    features.redirect_detected = originalUrl !== finalUrl;
    features.status_code = statusCode;
    features.extraction_timestamp = new Date().toISOString();
    features.user_agent = await page.evaluate(() => navigator.userAgent);
    features.viewport = await page.viewport();
    features.screenshot_path = screenshotPath;
    features.screenshot_taken = screenshotTaken;

    // Display results
    console.log(JSON.stringify(features));

    await browser.close();
})();

function normalizeUrl(inputUrl) {
    if (!inputUrl.startsWith('http://') && !inputUrl.startsWith('https://')) {
        return ['https://' + inputUrl, 'http://' + inputUrl];
    }
    return [inputUrl];
}

/**
 * Calculate average number of words per sentence in given text
 * @param {string} text - The text to analyze
 * @returns {number} Average words per sentence, 0 if no sentences found
 */
function calculateAvgWordsPerSentence(text) {
    if (!text || typeof text !== 'string') {
        return 0;
    }

    // Clean the text: remove extra whitespace and normalize
    const cleanText = text.trim().replace(/\s+/g, ' ');

    if (cleanText.length === 0) {
        return 0;
    }

    // Split into sentences using multiple delimiters
    // Look for periods, exclamation marks, question marks followed by space or end of string
    const sentences = cleanText
        .split(/[.!?]+\s+|[.!?]+$/)
        .filter(sentence => {
            // Filter out empty sentences and very short fragments (< 3 characters)
            const trimmed = sentence.trim();
            return trimmed.length >= 3 && /[a-zA-Z]/.test(trimmed);
        });

    if (sentences.length === 0) {
        return 0;
    }

    // Count words in each sentence
    let totalWords = 0;
    sentences.forEach(sentence => {
        // Split by whitespace and count non-empty words
        const words = sentence.trim().split(/\s+/).filter(word => {
            // Only count actual words (containing letters or numbers)
            return word.length > 0 && /[a-zA-Z0-9]/.test(word);
        });
        totalWords += words.length;
    });

    return totalWords / sentences.length;
}

/**
 * Calculate Flesch Reading Ease Score
 * Formula: 206.835 - (1.015 × ASL) - (84.6 × ASW)
 * Where: ASL = Average Sentence Length, ASW = Average Syllables per Word
 *
 * Score interpretation:
 * 90-100: Very Easy (5th grade level)
 * 80-89: Easy (6th grade level)
 * 70-79: Fairly Easy (7th grade level)
 * 60-69: Standard (8th-9th grade level)
 * 50-59: Fairly Difficult (10th-12th grade level)
 * 30-49: Difficult (College level)
 * 0-29: Very Difficult (Graduate level)
 *
 * @param {string} text - The text to analyze
 * @returns {number} Flesch Reading Ease Score (0-100+, can exceed 100 for very simple text)
 */
function calculateFleschScore(text) {
    if (!text || typeof text !== 'string') {
        return 0;
    }

    // Clean the text
    const cleanText = text.trim().replace(/\s+/g, ' ');

    if (cleanText.length === 0) {
        return 0;
    }

    // Count sentences
    const sentences = cleanText
        .split(/[.!?]+\s+|[.!?]+$/)
        .filter(sentence => {
            const trimmed = sentence.trim();
            return trimmed.length >= 3 && /[a-zA-Z]/.test(trimmed);
        });

    if (sentences.length === 0) {
        return 0;
    }

    // Count words
    const words = cleanText
        .split(/\s+/)
        .filter(word => {
            return word.length > 0 && /[a-zA-Z0-9]/.test(word);
        });

    if (words.length === 0) {
        return 0;
    }

    // Count syllables
    let totalSyllables = 0;
    words.forEach(word => {
        totalSyllables += countSyllables(word);
    });

    // Calculate metrics
    const averageSentenceLength = words.length / sentences.length;
    const averageSyllablesPerWord = totalSyllables / words.length;

    // Apply Flesch formula
    const fleschScore = 206.835 - (1.015 * averageSentenceLength) - (84.6 * averageSyllablesPerWord);

    // Round to 2 decimal places
    return Math.round(fleschScore * 100) / 100;
}

/**
 * Count syllables in a word using heuristic rules
 * This is an approximation as perfect syllable counting requires phonetic analysis
 * @param {string} word - The word to count syllables for
 * @returns {number} Estimated number of syllables
 */
function countSyllables(word) {
    if (!word || typeof word !== 'string') {
        return 0;
    }

    // Convert to lowercase and remove non-alphabetic characters
    word = word.toLowerCase().replace(/[^a-z]/g, '');

    if (word.length === 0) {
        return 0;
    }

    // Handle special cases
    if (word.length <= 3) {
        return 1;
    }

    // Count vowel groups
    let syllableCount = 0;
    let previousWasVowel = false;
    const vowels = 'aeiouy';

    for (let i = 0; i < word.length; i++) {
        const isVowel = vowels.includes(word[i]);

        if (isVowel && !previousWasVowel) {
            syllableCount++;
        }

        previousWasVowel = isVowel;
    }

    // Handle silent 'e' at the end
    if (word.endsWith('e') && syllableCount > 1) {
        syllableCount--;
    }

    // Handle special endings that add syllables
    const specialEndings = ['ed', 'es', 'er', 'le'];
    for (const ending of specialEndings) {
        if (word.endsWith(ending) && word.length > ending.length) {
            const beforeEnding = word.slice(0, -ending.length);
            if (ending === 'ed' && !vowels.includes(beforeEnding[beforeEnding.length - 1])) {
                // Past tense 'ed' after consonant usually doesn't add syllable
                continue;
            } else if (ending === 'le' && !vowels.includes(beforeEnding[beforeEnding.length - 1])) {
                // 'le' after consonant adds a syllable
                syllableCount++;
            }
        }
    }

    // Every word has at least one syllable
    return Math.max(1, syllableCount);
}
