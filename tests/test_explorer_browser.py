"""Rendered explorer regressions. Install Selenium and Chrome to run these locally."""
import os
import shutil

import pytest

pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait

from src.common import paths


@pytest.fixture(scope="module")
def browser():
    binary = next((path for name in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(name))), None)
    if binary is None:
        pytest.skip("Chrome or Chromium is required for browser tests")
    options = webdriver.ChromeOptions()
    options.binary_location = binary
    for arg in ("--headless=new", "--disable-dev-shm-usage", "--window-size=1510,960"):
        options.add_argument(arg)
    service = Service(os.environ.get("SE_CHROMEDRIVER") or shutil.which("chromedriver"))
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": ["http://*", "https://*"]})
        yield driver
    finally:
        driver.quit()


@pytest.mark.parametrize("mode", ["real", "dummy"])
def test_bundled_map_renders_offline(browser, mode):
    directory = paths.EXPLORER / "dummy" if mode == "dummy" else paths.EXPLORER
    browser.get((directory / "index.html").as_uri())
    WebDriverWait(browser, 20).until(lambda d: d.execute_script(
        "return document.querySelectorAll('#map .choroplethlayer path').length === Object.keys(tsData).length"))
    assert ("synthetic" in browser.find_element(By.ID, "header").text.lower()) == (mode == "dummy")
    if mode == "dummy":
        return

    scope = Select(browser.find_element(By.ID, "scope-select"))
    for window in ("india_core", "india_ext5"):
        scope.select_by_value(window)
        WebDriverWait(browser, 20).until(lambda d: d.execute_script("""
            const paths = document.querySelectorAll('#map .choroplethlayer path');
            if (paths.length !== indiaData.geojson.features.length) return false;
            const box = document.getElementById('map').getBoundingClientRect();
            return [...paths].every(path => {
                const r = path.getBoundingClientRect();
                return r.width < box.width * 0.8 || r.height < box.height * 0.8;
            });
        """))
    browser.find_element(By.ID, "tab-india").click()
    Select(browser.find_element(By.ID, "ind-rob-view")).select_by_value("kharif")
    assert browser.execute_script("""
        const trace = document.getElementById('ind-rob-chart').data[0];
        const rows = [...indiaData.stats.kharif, ...indiaData.stats.kharif_crops];
        return trace.x.length > 0 && trace.x.every((x, i) => {
            const row = rows.find(r => r.model + ' [' + r.term + ']' === trace.y[i]);
            return row && x === row.beta &&
                Math.abs(trace.error_x.array[i] - (row.ci_hi - x)) < 1e-13 &&
                Math.abs(trace.error_x.arrayminus[i] - (x - row.ci_lo)) < 1e-13;
        });
    """)
