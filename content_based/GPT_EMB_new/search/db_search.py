import pymysql

class DatabaseHandler:
    def __init__(self, host, port, user, password, database, charset):
        
        self.db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": charset
        }
        
        self.connection = None
        self.cursor = None
        
    def connect(self):
        
        try:
            self.connection = pymysql.connect(**self.db_config)
            self.cursor = self.connection.cursor()
        except Exception as e:
            print('Error connecting to database: ', e) 
            raise e

    def fetch_all_departments(self):
        """
        Fetch all department IDs and names from the department table.
        """
        try:
            if not self.connection or not self.cursor:
                raise Exception("Database connection is not established.")

            # SQL query to fetch all department IDs and names
            sql_query = """
            SELECT 
                id AS department_id,
                name AS department_name
            FROM jbnu_department
            ORDER BY id;
            """
            self.cursor.execute(sql_query)

            # Fetch column names
            columns = [col[0] for col in self.cursor.description]

            # Fetch results and convert to list of dictionaries
            rows = self.cursor.fetchall()
            print(f"{len(rows)} departments fetched.")
            return [dict(zip(columns, row)) for row in rows]

        except pymysql.MySQLError as e:
            print(f"Error executing query: {e}")
            raise
    
    #deprtment goal
    def fetch_depart_goal(self):
        """
        Fetch all department IDs and names from the department table.
        """
        try:
            if not self.connection or not self.cursor:
                raise Exception("Database connection is not established.")
            
            sql_query = """
                SELECT
                    jc.id AS class_id,
                    jc.name AS class_name,
                    jc.description AS class_description,
                    jc.student_grade AS class_student_grade,
                    jd.name AS department_name,
                    jc.curriculum,
                    jd.id AS department_id,
                    jco.name AS college_name
                FROM jbnu_class jc
                JOIN jbnu_department jd ON jc.department_id = jd.id
                JOIN jbnu_college jco ON jd.college_id = jco.id
                WHERE jc.student_grade IN (3,4) -- Filter for 3rd and 4th year students
                ORDER BY jc.id;
            """
            self.cursor.execute(sql_query)
            columns = [col[0] for col in self.cursor.description]
            rows = self.cursor.fetchall()
            results = [dict(zip(columns, row)) for row in rows]
            print(f'{len(rows)} departments fetched.')
            return results
        
        except pymysql.MySQLError as e:
            print(f"Error executing query: {e}")
            raise
            
            
    def fetch_classes_by_department(self, department_id):
        """
        Fetch class data along with department and college details for a specific department_id.
        """
        try:
            if not self.connection or not self.cursor:
                raise Exception("Database connection is not established.")

            # SQL query with department_id filter
            sql_query = """
                SELECT
                    jc.id,
                    jc.name,
                    jc.description,
                    jc.student_grade,
                    jd.name,
                    jco.name
                FROM jbnu_class jc
                JOIN jbnu_department jd ON jc.department_id = jd.id
                JOIN jbnu_college jco ON jd.college_id = jco.id
                WHERE jc.student_grade IN (3, 4) -- 3학년과 4학년 필터링
                ORDER BY jc.id;
            """
            
            self.cursor.execute(sql_query, (department_id,))

            # Fetch column names
            columns = [col[0] for col in self.cursor.description]

            # Fetch results and convert to list of dictionaries
            rows = self.cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

        except pymysql.MySQLError as e:
            print(f"Error executing query: {e}")
            raise
    

    def fetch_classes_info_by_department(self):
        """
        Fetch class data along with department and college details for a specific department.
        """
        
        try:
            if not self.connection or not self.cursor:
                raise Exception("Database connection is not established.")
            
            sql_query = """
                SELECT
                    jc.id AS class_id,
                    jc.name AS class_name,
                    jc.description AS class_description,
                    jc.curriculum AS class_curriculum,
                    jc.content AS class_content,
                    jc.student_grade AS student_grade,
                    jd.name AS department_name,
                    jd.id AS department_id,
                    jco.name AS college_name
                FROM jbnu_class jc
                JOIN jbnu_department jd ON jc.department_id = jd.id
                JOIN jbnu_college jco ON jd.college_id = jco.id
                ORDER BY jc.student_grade, jc.id;
            """
            self.cursor.execute(sql_query)
            
            #coulums name
            columns = [col[0] for col in self.cursor.description]
            rows = self.cursor.fetchall()
            results = [dict(zip(columns, row)) for row in rows]
            return results
        
        except pymysql.MySQLError as e:
            print(f"Error executing query: {e}")
            raise
        
    
    # hy (추가)
    # 🔹 특정 단대를 제외한 강의 데이터 가져오기 (학과 정보 포함)
    def fetch_filtered_classes(self):
        try:
            if not self.connection or not self.cursor:
                raise Exception("Database connection is not established.")
        
            query = """
            SELECT c.id, c.name, c.description, c.curriculum, c.student_grade, c.semester, d.name AS department_name
            FROM jbnu_class c
            JOIN jbnu_department d ON c.department_id = d.id
            JOIN jbnu_college col ON d.college_id = col.id
            WHERE col.name NOT IN ('사범대학', '예술대학', '글로벌융합대학')
            """
            
            self.cursor.execute(query)
        
            # 컬럼명 가져오기
            columns = [col[0] for col in self.cursor.description]
            rows = self.cursor.fetchall()

            # 결과를 딕셔너리 리스트로 변환
            results = [dict(zip(columns, row)) for row in rows]

            # `full_text` 필드 추가 (description이 None일 경우 빈 문자열 처리)
            for item in results:
                item["full_text"] = f"{item['name']} {item['description'] or ''}"

            return results
        
        except pymysql.MySQLError as e:
            print(f"Error executing query: {e}")
            raise
    
    
    # 🔹 특정 학과(들)에 대해서만 강의 데이터를 가져오기
    def fetch_filtered_classes_dept(self, department_names=None):
        try:
            if not self.connection or not self.cursor:
                raise Exception("Database connection is not established.")
        
            query = """
            SELECT c.id, c.name, c.description, c.curriculum, c.student_grade, c.semester, d.name AS department_name
            FROM jbnu_class c
            JOIN jbnu_department d ON c.department_id = d.id
            JOIN jbnu_college col ON d.college_id = col.id
            WHERE col.name NOT IN ('사범대학', '예술대학', '글로벌융합대학')
            """
            
            self.cursor.execute(query)
        
            # 컬럼명 가져오기
            columns = [col[0] for col in self.cursor.description]
            rows = self.cursor.fetchall()

            # 결과를 딕셔너리 리스트로 변환
            results = [dict(zip(columns, row)) for row in rows]

            # 🔹 특정 학과 필터링 적용
            if department_names:
                results = [item for item in results if item["department_name"] in department_names]

            # 🔹 `full_text` 필드 추가 (강의명 + 설명)
            for item in results:
                item["full_text"] = f"{item['name']} {item['description'] or ''}"

            return results

        except pymysql.MySQLError as e:
                print(f"Error executing query: {e}")
                raise

    def close(self):
        if self.connection:
            self.cursor.close()
            self.connection.close()
            self.connection = None
            self.cursor = None
        else:
            raise Exception('Database connection not established.')