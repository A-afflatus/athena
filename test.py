import asyncio
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

class SearchBilibiliVideosInput(BaseModel):
    """搜索bilibili视频输入"""
    keywords: str = Field(description="搜索关键词")
async def search_bilibili_videos(input: SearchBilibiliVideosInput):
    results = []
    async with async_playwright() as p:
        # 启动浏览器，使用无头模式
        browser = await p.chromium.launch(headless=False)
        # 模拟真实的浏览器环境，设置 User-Agent
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        url = f"https://search.bilibili.com/all?keyword={input.keywords}"
        try:
            # 访问页面，等待网络空闲
            await page.goto(url, wait_until="networkidle", timeout=30000)
            # 等待视频列表加载
            await page.wait_for_selector(".video-list", timeout=10000)

            # 遍历 .video-list 下的儿子 div 元素
            video_elements = await page.query_selector_all(".video-list > div")

            for element in video_elements:
                # 提取标题
                title_el = await element.query_selector(".bili-video-card__info--tit")
                # 提取链接
                link_el = await element.query_selector(":scope > a")
                # 提取作者
                author_el = await element.query_selector(".bili-video-card__info--author")

                # image url
                # image_el = await element.query_selector("img") # 找alt和title相等的img标签的src并补全https:
                # 视频时长
                duration_el = await element.query_selector(".bili-video-card__stats__duration")
                # 发布日期
                date_el = await element.query_selector(".bili-video-card__info--date")
                # 播放量
                play_el = await element.query_selector(".bili-video-card__stats--item > span")

                if title_el and link_el:
                    author = await author_el.inner_text() if author_el else "未知"
                    title = await title_el.get_attribute("title") or await title_el.inner_text()
                    link = f"https:{await link_el.get_attribute("href")}"
                    duration = await duration_el.inner_text() if duration_el else None
                    date = await date_el.inner_text() if date_el else None
                    play = await play_el.inner_text() if play_el else None
                    results.append(
                        {
                            "title": title.strip(),
                            "link": link,
                            "author": author.strip(),
                            "duration": duration,
                            "date": date,
                            "play": play,
                        }
                    )

                if len(results) >= 5:  # 返回前 5 条结果即可
                    break

            if not results:
                content = f"搜索关键词 '{input.keywords}' 未找到视频结果。"
            else:
                res_str = "\n".join(
                    [
                        f"{i+1}. {r['title']} (UP: {r['author']}) - {r['link']} - 时长: {r['duration']} - 发布日期: {r['date']} - 播放量: {r['play']}"
                        for i, r in enumerate(results)
                    ]
                )
                content = f"为你找到以下关于 '{input.keywords}' 的 Bilibili 视频：\n{res_str}"
            print(content)
        except Exception as e:
            content = f"在 Bilibili 搜索视频时发生错误: {str(e)}"
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(search_bilibili_videos(SearchBilibiliVideosInput(keywords="Python")))