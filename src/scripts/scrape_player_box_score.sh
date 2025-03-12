echo "Starting player scraping"

python src/scraping/async_player_box_score_scraper.py --sport men --division d1 --year 2021
python src/scraping/async_player_box_score_scraper.py --sport men --division d1 --year 2022
python src/scraping/async_player_box_score_scraper.py --sport men --division d1 --year 2023
python src/scraping/async_player_box_score_scraper.py --sport men --division d1 --year 2024

python src/scraping/async_player_box_score_scraper.py --sport women --division d1 --year 2021
python src/scraping/async_player_box_score_scraper.py --sport women --division d1 --year 2022
python src/scraping/async_player_box_score_scraper.py --sport women --division d1 --year 2023
python src/scraping/async_player_box_score_scraper.py --sport women --division d1 --year 2024
