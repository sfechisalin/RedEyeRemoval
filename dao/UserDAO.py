import mysql.connector
import configparser
from mysql.connector import Error
from mysql.connector import errorcode

class UserDAO:
    def __init__(self):
        self.__connection = self.__get_connection()

    def __get_connection(self):
        try:
            config = configparser.RawConfigParser()
            config.read('./dao/config.properties')

            host = config.get('DatabaseSection', 'database.host')
            database_name = config.get('DatabaseSection', 'database.dbname')
            username = config.get('DatabaseSection', 'database.user')
            password = config.get('DatabaseSection', 'database.password')

            print(host, database_name, username, password)
            connection = mysql.connector.connect(host=host,
                                                 database=database_name,
                                                 user=username,
                                                 password=password,
                                                 use_pure=True)
            print("Connection done successfully")
            return connection
        except mysql.connector.Error as error:
            #self.__connection.rollback()
            print("Error: {}".format(error))
            return None

    def exists_user(self, username, password):
        self.__connection = self.__get_connection()

        if self.__connection != None:
           try:
                cursor = self.__connection.cursor(prepared=True)
                sql_select_query = """select * from USERS WHERE username=%s AND password=%s"""
                input = (username, password)
                cursor.execute(sql_select_query, input)
                records = cursor.fetchall()
                if len(records) > 0:
                    return True
                else:
                    return False
           except mysql.connector.Error as error:
               self.__connection.rollback()
               print("Error: {}".format(error))
           finally:
               if self.__connection.is_connected():
                   cursor.close()
                   self.__connection.close()
                   print("MySQL connection is closed")
        else:
            print("Connection could not be established")
            return False

    def add_user(self, username, password):
            if self.exists_user(username, password) == False:
                try:
                    self.__connection = self.__get_connection()
                    cursor = self.__connection.cursor(prepared=True)
                    sql_insert_query = """ INSERT INTO `USERS`
                                             (`username`, `password`, `number_of_attempts`) VALUES (%s,%s,%s)"""
                    insert_tuple = (username, password, 0)
                    result = cursor.execute(sql_insert_query, insert_tuple)
                    self.__connection.commit()
                    print("Record inserted successfully into python_users table")
                    return True
                except mysql.connector.Error as error:
                    self.__connection.rollback()
                    print("Failed to insert into MySQL table {}".format(error))
                    return False
                finally:
                    if self.__connection.is_connected():
                        cursor.close()
                        self.__connection.close()
                        print("MySQL connection is closed")
            else:
                return False

    def get_attempts(self, username):
        self.__connection = self.__get_connection()

        if self.__connection != None:
            try:
                cursor = self.__connection.cursor(prepared=True)
                sql_select_query = """select number_of_attempts from USERS WHERE username=%s"""
                cursor.execute(sql_select_query, (username, ))
                records = cursor.fetchone()
                if len(records) > 0:
                    no = int(records[0])
                    return no
                else:
                    return None
            except mysql.connector.Error as error:
                self.__connection.rollback()
                print("Error: {}".format(error))
            finally:
                if self.__connection.is_connected():
                    cursor.close()
                    self.__connection.close()
                    print("MySQL connection is closed")
        else:
            print("Connection could not be established")
            return False

    def update_attempts(self, username, no_new_attempts):
        self.__connection = self.__get_connection()

        if self.__connection != None:
            try:
                cursor = self.__connection.cursor(prepared=True)
                sql_update_query = """UPDATE USERS SET number_of_attempts = %s WHERE username = %s"""
                insert_tuple = (no_new_attempts, username)
                cursor.execute(sql_update_query, insert_tuple)
                self.__connection.commit()
                print("Record Updated successfully with prepared statement")
                return True
            except mysql.connector.Error as error:
                self.__connection.rollback()
                print("Failed to update record to database: {}".format(error))
            finally:
                if self.__connection.is_connected():
                    cursor.close()
                    self.__connection.close()
                    print("MySQL connection is closed")
        else:
            print("Connection could not be established")
        return False