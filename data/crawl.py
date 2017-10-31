# coding: utf-8

from github import Github
import argparse
import base64

g = Github('username', 'passsword') # enter username and password for larger query limits

parser = argparse.ArgumentParser(description='Download README.md from public repositories on Github')

parser.add_argument('-n', type=int, default=1000,
                    help='# of README files to download')
parser.add_argument('--keyword', type=str, default='javascript',
                    help='keyword used to search public repos')
parser.add_argument('--limit', type=int, default=300,
                    help='minimum # of words in a valid README file')
parser.add_argument('-o', '--output', type=str, default='crawl.txt',
                    help='output path')

args = parser.parse_args()

count = 0
outfile = open(args.output, 'w', encoding='utf8')
print_every = 100

for repo in g.search_repositories('javascript', sort='updated'):
    try:
        readme = repo.get_readme()
        content = base64.b64decode(readme.content)

        if len(content) > args.limit:
            count += 1
            print(content, file=outfile)

            if count % print_every == 0:
                print(count)

            if count >= args.n:
                break

    except Exception as e:
        print(e)
    
outfile.close()