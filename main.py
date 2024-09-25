import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configurar o driver do Chrome
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Abrir o site desejado
url = "https://eproc1g.tjrs.jus.br"
driver.get(url)

# Esperar até que o campo de login esteja visível
wait = WebDriverWait(driver, 10)
login_field = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="txtUsuario"]')))

# Preencher o campo de login
login_field.send_keys("AACFGM*1970")

# Localizar e preencher o campo de senha
password_field = driver.find_element(By.XPATH, '//*[@id="pwdSenha"]')
password_field.send_keys("GE2TGYRTMMYDENZRMEZTKM3BGI4TCZTC")

clicar_no_buuton = driver.find_element(By.XPATH, '//*[@id="sbmEntrar"]')
clicar_no_buuton.click()

i = 300

for i in range(500):
    driver.find_element(By.XPATH, '//*[@id="txtInfraCaptcha"]').send_keys("a")
    driver.find_element(By.XPATH, '//*[@id="frmLogin"]/div[2]/button').click()
    time.sleep(1)
    driver.find_element(By.XPATH, '/html/body/div/div[3]/div[2]/div/div/div[2]/div/div[1]/div/div/form/div[1]/label/img').screenshot(f'captcha_solver\\datasets\\capchasolver{i}.png')
    i =+ 1
# Aguardar um pouco para ver o resultado (você pode ajustar ou remover este tempo de espera)
time.sleep(5)

# Fechar o navegador (descomente quando quiser fechar automaticamente)
# driver.quit()
