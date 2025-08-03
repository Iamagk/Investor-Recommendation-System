import asyncio
from playwright.async_api import async_playwright

async def scrape_all_mutual_funds():
    url = "https://www.etmoney.com/mutual-funds/funds"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, timeout=60000)
        await page.wait_for_selector(".fund-listing__row", timeout=60000)

        # Scroll to load all funds (ET Money loads dynamically)
        for _ in range(20):  # Adjust number of scrolls as needed
            await page.mouse.wheel(0, 10000)
            await asyncio.sleep(1)

        rows = await page.query_selector_all(".fund-listing__row")
        data = []

        for row in rows:
            name = await row.query_selector(".fund-name__name")
            category = await row.query_selector(".fund-name__category")
            returns = await row.query_selector(".fund-performance__value")
            
            fund = {
                "name": await name.inner_text() if name else None,
                "category": await category.inner_text() if category else None,
                "returns": await returns.inner_text() if returns else None
            }
            data.append(fund)

        await browser.close()
        return data