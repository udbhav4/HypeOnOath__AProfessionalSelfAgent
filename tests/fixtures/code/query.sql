-- Several SQL statements: no functions, size-based chunks, empty symbols
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO users (id, name) VALUES (1, 'a'), (2, 'b');
SELECT id, name FROM users WHERE id > 1 ORDER BY name;
UPDATE users SET name = 'c' WHERE id = 2;
DELETE FROM users WHERE id = 1;
