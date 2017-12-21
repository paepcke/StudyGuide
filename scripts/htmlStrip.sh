#replace file.txt with the file name to strip html tags and new.txt with new file name
sed -e 's/<[^>]*>//g' file.txt > new.txt
