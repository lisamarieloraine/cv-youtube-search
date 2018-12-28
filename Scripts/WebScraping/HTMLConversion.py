# Reference:
# https://www.scrapehero.com/scrape-yahoo-finance-stock-market-data/

from lxml import html
import xml.etree.ElementTree as ET


# @argument <class 'bytes'>
# @return <class 'lxml.html.HtmlElement'>
def response_to_html(_response_data):
    """Returns html from url response"""
    return html.fromstring(_response_data)


# @argument <class 'bytes'>
# @return <class 'xml.etree.ElementTree.Element'>
def response_to_xml(_response_data):
    """Returns xml from url response"""
    return ET.fromstring(_response_data)


# @argument <class 'lxml.html.HtmlElement'>
# @return <class 'list'> containing <class 'lxml.html.HtmlElement'>
def get_lxml_element(_html, _xpath):
    """Returns filtered html from xpath"""
    return _html.xpath(_xpath)
