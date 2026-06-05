-- 01-setup.sql
-- Настройка прав для Nextcloud

-- Сначала убедимся, что база данных существует
-- Если нет - создаем ее

DO $$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'nextcloud') THEN
            CREATE USER nextcloud WITH SUPERUSER PASSWORD 'nextcloud123!@';
        ELSE
            RAISE NOTICE 'Пользователь nextcloud уже существует';
        END IF;
    END
$$;
SELECT 'CREATE DATABASE nextcloud OWNER nextcloud'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'nextcloud')\gexec;


-- Подключаемся к базе данных
\c nextcloud


-- Создаем схему public (если не существует)
CREATE SCHEMA IF NOT EXISTS public;

-- Даем все права на схему public
GRANT ALL ON SCHEMA public TO nextcloud;
GRANT ALL PRIVILEGES ON SCHEMA public TO nextcloud;
GRANT CREATE ON SCHEMA public TO nextcloud;

-- Делаем пользователя владельцем схемы
ALTER SCHEMA public OWNER TO nextcloud;

-- Устанавливаем search_path для базы данных
ALTER DATABASE nextcloud SET search_path TO public;

-- Устанавливаем права по умолчанию для новых таблиц
ALTER DEFAULT PRIVILEGES FOR USER nextcloud IN SCHEMA public
    GRANT ALL ON TABLES TO nextcloud;
ALTER DEFAULT PRIVILEGES FOR USER nextcloud IN SCHEMA public
    GRANT ALL ON SEQUENCES TO nextcloud;
ALTER DEFAULT PRIVILEGES FOR USER nextcloud IN SCHEMA public
    GRANT ALL ON FUNCTIONS TO nextcloud;

-- Создаем необходимые расширения (если нужны)
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Выводим информацию для отладки
\du
\dn+
