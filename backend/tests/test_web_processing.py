from __future__ import annotations

import unittest

from app.processing.web import WebExtractionError, extract_web_page, select_attachments, selected_attachments


HTML = """
<html>
  <head><title>资助通知</title></head>
  <body>
    <nav>热点阅读 这是导航，不是正文</nav>
    <main>
      <h1>资助通知</h1>
      <p>学校发布本年度学生资助政策说明，学生可以按照通知准备材料。</p>
      <p><a href="/files/policy.pdf">资助政策 PDF</a></p>
      <p><a href="/files/form.docx">申请表 DOCX</a></p>
      <p><a href="/notice/other.html">普通相关链接</a></p>
    </main>
  </body>
</html>
"""


class WebProcessingTest(unittest.TestCase):
    def test_extracts_body_and_only_supported_attachments(self) -> None:
        page = extract_web_page(HTML, "https://example.com/news/notice.html")

        self.assertEqual(page.title, "资助通知")
        self.assertIn("学生资助政策说明", page.body_text)
        self.assertNotIn("热点阅读", page.body_text)
        self.assertEqual(len(page.attachments), 2)
        self.assertEqual(page.attachments[0].url, "https://example.com/files/policy.pdf")
        self.assertEqual(page.attachments[0].file_type, "PDF")
        self.assertEqual(page.attachments[1].file_type, "Word")

    def test_attachment_selection_is_explicit(self) -> None:
        page = extract_web_page(HTML, "https://example.com/news/notice.html")
        selected = select_attachments(page.attachments, [page.attachments[1].url])

        self.assertEqual(len(selected_attachments(selected)), 1)
        self.assertEqual(selected_attachments(selected)[0].file_type, "Word")
        with self.assertRaises(WebExtractionError):
            select_attachments(page.attachments, ["https://example.com/files/other.xlsx"])

    def test_navigation_only_page_is_rejected(self) -> None:
        with self.assertRaises(WebExtractionError):
            extract_web_page("<html><body><nav>热点阅读</nav></body></html>", "https://example.com")


if __name__ == "__main__":
    unittest.main()
